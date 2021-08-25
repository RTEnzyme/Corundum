#![cfg(feature = "cbindings")]

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ffi::c_void;
use std::panic::{UnwindSafe, RefUnwindSafe};
use std::mem::size_of;
use crate::*;
use crate::stm::Journal;
use crate::clone::PClone;
use crate::alloc::*;
use crate::ptr::*;
use crate::stm::{Logger,Notifier};

#[repr(C)]
pub struct Gen<T, P: MemPool> {
    ptr: *const c_void,
    len: usize,
    phantom: PhantomData<(T,P)>
}

unsafe impl<T, P: MemPool> TxInSafe for Gen<T, P> {}
unsafe impl<T, P: MemPool> LooseTxInUnsafe for Gen<T, P> {}
impl<T, P: MemPool> UnwindSafe for Gen<T, P> {}
impl<T, P: MemPool> RefUnwindSafe for Gen<T, P> {}

/// A byte-vector representation of any type
/// 
/// It is useful for FFI functions when template types cannot be externally used.
/// 
/// # Examples
/// 
/// ```
/// corundum::pool!(pool);
/// use pool::*;
/// type P = Allocator;
/// 
/// use corundum::gen::{ByteArray,Gen};
/// 
/// struct ExternalType {
///     obj: ByteArray<P>
/// }
/// 
/// #[no_mangle]
/// pub extern "C" fn new_obj(obj: Gen) {
///     
/// }
/// ```
#[derive(Clone)]
pub struct ByteArray<P: MemPool> {
    bytes: Slice<u8, P>,
    logged: u8
}

// impl<P: MemPool> Copy for ByteArray<P> {}

unsafe impl<P: MemPool> PSafe for ByteArray<P> {}
unsafe impl<P: MemPool> LooseTxInUnsafe for ByteArray<P> {}
impl<P: MemPool> UnwindSafe for ByteArray<P> {}
impl<P: MemPool> RefUnwindSafe for ByteArray<P> {}

impl<P: MemPool> Default for ByteArray<P> {
    fn default() -> Self {
        Self {
            bytes: Default::default(),
            logged: 0
        }
    }
}

impl<P: MemPool> PClone<P> for ByteArray<P> {
    fn pclone(&self, j: &Journal<P>) -> Self {
        Self {
            bytes: self.bytes.pclone(j),
            logged: 0
        }
    }
}

impl<P: MemPool> Drop for ByteArray<P> {
    fn drop(&mut self) {
        unsafe {
            if !self.bytes.is_empty() {
                P::dealloc(self.bytes.as_mut_ptr(), self.bytes.capacity())
            }
        }
    }
}

impl<P: MemPool> ByteArray<P> {
    pub fn new_uninit(size: usize, j: &Journal<P>) -> Self {
        unsafe {
            let ptr = P::new_uninit_for_layout(size, j);
            Self { bytes: Slice::from_raw_parts(ptr, size), logged: 0 }
        }
    }

    pub fn null() -> Self {
        Self {
            bytes: Default::default(),
            logged: 0
        }
    }

    pub fn move_from(&mut self, other: &Self) {
        let other = unsafe { utils::as_mut(other) };
        self.bytes = other.bytes;
        self.logged = other.logged;
        other.bytes = Slice::null();
        other.logged = 0;
    }

    // pub fn as_bytes(&self) -> Vec<u8> {
    //     self.bytes.to_vec()
    // }

    pub fn as_ref<T>(&self) -> &T {
        unsafe { &*(self.bytes.as_ptr() as *const T) }
    }

    pub fn from_gen<T>(obj: Gen<T, P>, j: &Journal<P>) -> Self {
        let bytes = obj.as_slice();
        unsafe {
            let bytes = P::new_slice(bytes, j);
            Self { bytes: Slice::new(bytes), logged: 0 }
        }
    }

    pub fn as_gen<T>(self) -> Gen<T, P> {
        Gen::from_byte_object(self)
    }

    pub unsafe fn from_ref_gen<T>(mut obj: Gen<T, P>) -> Self {
        let bytes = obj.as_slice_mut();
        Self { bytes: Slice::from_raw_parts(bytes.as_mut_ptr(), bytes.len()), logged: 0 }
    }

    pub unsafe fn as_ref_gen<T>(&self) -> Gen<T, P> {
        // assert_eq!(self.len(), size_of::<T>(), "Incompatible type casting");
        Gen::<T, P>::from_ptr(self.as_ptr::<T>())
    }

    pub unsafe fn as_mut<T>(&self) -> &mut T {
        &mut *(self.bytes.as_ptr() as *mut T)
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.bytes.as_ptr() as *const T
    }

    pub fn as_ptr_mut(&mut self) -> *mut c_void {
        self.bytes.as_ptr() as *mut c_void
    }

    pub unsafe fn to_ptr_mut(slf: &mut Self) -> *mut c_void {
        slf.bytes.as_ptr() as *mut c_void
    }

    pub fn off(&self) -> u64 {
        self.bytes.off()
    }

    pub fn write_to<T>(&self, loc: &mut MaybeUninit<T>) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.bytes.as_ptr(), 
                loc as *mut _ as *mut u8, 
                self.bytes.capacity());
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.capacity()
    }

    pub fn update_from_gen<T>(&self, new: Gen<T, P>, j: &Journal<P>) {
        unsafe {
            let slice = crate::utils::as_mut(self).bytes.as_slice_mut();
            if self.logged == 0 {
                slice.create_log(j, Notifier::NonAtomic(Ptr::from_ref(&self.logged)));
            }
            std::ptr::copy_nonoverlapping(new.ptr, slice as *mut [u8] as *mut c_void, slice.len())
        }
    }
}

impl<T, P: MemPool> Gen<T, P> {
    pub fn null() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
            // drop: false,
            phantom: PhantomData
        }
    }
}

impl<T, P: MemPool> Gen<T, P> {
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *mut u8, self.len) }
    }

    fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    fn from_ptr(obj: *const T) -> Self {
        Self {
            ptr: obj as *const T as *const c_void,
            len: size_of::<T>(),
            // drop: false,
            phantom: PhantomData
        }
    }

    fn from_byte_object(obj: ByteArray<P>) -> Self {
        // assert_eq!(obj.len(), size_of::<T>(), "Incompatible type casting");
        Self {
            ptr: obj.as_ptr(),
            len: obj.len(),
            phantom: PhantomData
        }
    }

    pub fn ptr(&self) -> *const c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;

//     impl<T, P: MemPool> From<&T> for Gen<T, P> {
//         fn from(obj: &T) -> Self {
//             Self {
//                 ptr: obj as *const T as *const c_void,
//                 len: size_of::<T>(),
//                 phantom: PhantomData
//             }
//         }
//     }
// }