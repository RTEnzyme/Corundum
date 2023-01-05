use std::alloc::{Allocator, AllocError};
use std::cell::RefCell;
use std::fs::File;
use std::mem;
use std::path::PathBuf;
use std::ptr::NonNull;
use tempfile;
use memmap::MmapMut;
use tracing::{debug, info, span, Level, debug_span};

use crate::utils::read_addr;

#[derive(Debug, Clone)]
/// Buddy memory blocl
/// Each memory block has some meta-data information in form of `Buddy` data
/// structure. It has a pointer to the next buddy block, if there is any.
struct Buddy {
    /// Next pointer offset
    /// We assume that usize::MAX is NULL
    next: u64,
}

impl Default for Buddy {
    fn default() -> Self {
        Self { next: u64::MAX }
    }
}

#[inline]
fn is_none(p: u64) -> bool {
    p == u64::MAX
}

#[inline]
fn off_to_option(p: u64) -> Option<u64> {
    if is_none(p) {
        None
    } else {
        Some(p)
    }
}

#[inline]
fn option_to_ppter(p: Option<u64>) -> u64 {
    if let Some(p) = p {
        p
    } else {
        u64::MAX
    }
}

// let mut Pmem_V_Allocator_Default = BuddyVolatileAlg::new(0, 1024*1084*16);

pub struct PmemVBuddyAllocator {
    /// the path /path/to/pmem_dir
    // pmem_path: PathBuf,
    tempfile: File,

    mmap: MmapMut,
    /// the device size
    size: usize,

    // the instance of BuddyVolatileAlg
    buddy_alg: RefCell<BuddyVolatileAlg>,
}

impl PmemVBuddyAllocator {
    pub fn new(path: PathBuf, size: usize) -> Self {
        info!("Instance a new PmemVBuddyAllocator");
        let tempfile = tempfile::tempfile_in(path).unwrap();
        debug!("Create a new tempfile");
        if let Some(err) = tempfile.set_len(size as u64).err() {
            panic!("{}\n create tempfile error when set file length: {}.", err, size);
        }
        let mut mmap = unsafe {
            memmap::MmapOptions::new().map_mut(&tempfile).unwrap()
        };

        let begin = mmap.get_mut(0).unwrap();
        // unsafe {
        //     std::ptr::write_bytes(begin, 0xff, 8);
        // }
        let base = begin as *const _ as u64;
        debug!("The base address of mapped tempfile: {}", base);
        // let base
        Self { tempfile, mmap, size, buddy_alg: RefCell::new(BuddyVolatileAlg::new(base, size)) }
    }
}

/// Implement the std::alloc::Allocator for Pmem_V_Buddy_Allocator
/// So, we can use Pmem_V_buddy_Allocator as many std data structs' Allocator
/// We can use std::Boxed<T, Pmem_V_Buddy_Allocator> to treat pmem as the large 
/// scalabity volatile memory.
/// 
unsafe impl Allocator for PmemVBuddyAllocator {
    fn allocate(&self, layout: std::alloc::Layout) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let span = span!(Level::DEBUG, "Allocate");
        let _enter = span.enter();
        debug!("Allocate {} bytes layout", layout.size());
        unsafe {
            let off = self.buddy_alg.borrow_mut().alloc_impl(layout.size());
            debug!("Allocate off: {}", off);
            let ptr = NonNull::new(off as *mut _).ok_or(AllocError)?;
            debug!("Allocate success!");
            Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
        }
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        debug!("free {} bytes layout", layout.size());
        unsafe {
            self.buddy_alg.borrow_mut().dealloc_impl(ptr.as_ptr() as u64, layout.size());
        }
    }

    fn allocate_zeroed(&self, layout: std::alloc::Layout) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let ptr = self.allocate(layout)?;
        // SAFETY: `alloc` returns a valid memory block
        unsafe { ptr.as_non_null_ptr().as_ptr().write_bytes(0, ptr.len()) }
        Ok(ptr)
    }

    unsafe fn grow(
        &self,
        ptr: std::ptr::NonNull<u8>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;

        // SAFETY: because `new_layout.size()` must be greater than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `old_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_layout.size());
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: std::ptr::NonNull<u8>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate_zeroed(new_layout)?;

        // SAFETY: because `new_layout.size()` must be greater than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `old_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_layout.size());
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    unsafe fn shrink(
        &self,
        ptr: std::ptr::NonNull<u8>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;

        // SAFETY: because `new_layout.size()` must be lower than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `new_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), new_layout.size());
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }
}

/// Budyy Algorithm for Pmem as an Dram Extension
/// 
/// It contains 64 free-lists of available buddy blocks to keep at most `2^64`
/// bytes including meta-data information. A free-list `k` keeps all available
/// memory blocks of size `2^k` bytes. Assuming that `Buddy` has a size of 
/// 8 bytes, the shape of lists can be like this:
/// 
/// ```text
///    [8]: [8] -> [8]
///   [16]: [8|8] -> [8|8]
///   [32]: [8|24] -> [8|24] -> [8|24]
///   [64]: [8|56]
///   ...
/// ```
/// 
/// The first 8 bytes of each free block is meta-data. Once they are selected
/// for occupation, this 8 bytes are goint to be used, too. So, the smallest block
/// size are 8 bytes.
struct BuddyVolatileAlg {
    /// Lists of free blocks
    buddies: [u64; 64],

    /// base address
    base: u64,

    /// end address
    end: u64,

    /// The index of the last buddy list
    last_idx: usize,

    /// Total available space in bytes
    available: usize,

    /// The device size in bytes
    size: usize,

    // A mutex for atomic operations
    // mutex: Mutex<()>,

    // Marker
    // phantom: PhantomData<A>,
}

#[inline]
const fn num_bits<T>() -> u32 {
    // Convert bytes to bits
    (mem::size_of::<T>() << 3) as u32
}

#[inline]
pub fn get_idx(x: usize) -> usize {
    if x == 0 {
        usize::MAX
    } else {
        // Make 8 bytes at least to record meta-data
        let x = x.max(mem::size_of::<Buddy>());
        (num_bits::<usize>() - (x - 1).leading_zeros()) as usize
    }
}

impl BuddyVolatileAlg {
    /// Pool Initialization with a given device size
    pub fn new(base: u64, size: usize) -> Self {
        let span = debug_span!("Buddy Volatile Alg");
        let _enter = span.enter();

        let mut idx = get_idx(size);
        debug!("intialize {idx:} idx buddy list");
        if 1 << idx > size {
            idx -= 1;
        } 
        let mut buddies = [u64::MAX; 64];
        let size = 1 << idx;
        let available = size;
        buddies[idx] = 0;
        let last_idx = idx;
        let end = base + size as u64;
        // let mutex = Mutex::new(());
        
        unsafe {
            read_addr::<Buddy>(base).next = u64::MAX;
        }

        unsafe {
            debug!("{:?}", read_addr::<u64>(base));
        }

        Self {
            buddies,
            base,
            end,
            last_idx,
            available,
            size,
            // mutex,
        }
    }

    #[inline]
    fn in_range<'a>(&self, off: u64) -> bool {
        (off < u64::MAX - self.base) && (off + self.base < self.end)
    }

    #[inline]
    #[track_caller]
    fn buddy<'a>(&self, off: u64) -> &'a mut Buddy {
        unsafe {read_addr(self.base + off)}
    }

    #[inline]
    fn byte<'a>(&self, off: u64) -> &'a mut u8 {
        unsafe { read_addr(self.base + off) }
    }

    #[inline]
    fn get_off(&self, b: &u64) -> u64 {
        let off = b as *const _ as u64;
        off - self.base
    }

    #[inline]
    unsafe fn find_free_memory(&mut self, idx: usize, split: bool) -> Option<u64> {
        debug!("find free memory in idx {idx:} and {split:} split.");
        if idx > self.last_idx {
            None
        } else {
            let res;
            if let Some(b) = off_to_option(self.buddies[idx]) {
                // Remove the available block and return it
                unsafe {
                    debug!("{}", self.base);
                    debug!("{:?}", read_addr::<u64>(self.base));
                }
                let buddy = self.buddy(b);
                debug!("remove one {idx:} idx block. The head of list is {}.", self.buddies[idx]);
                self.perform(self.buddies[idx], buddy.next);
                res = b;
            } else {
                res = self.find_free_memory(idx + 1, true)?;
            }
            if idx > 0 && split {
                // find the tail of the Buddy
                let next = res + (1 << (idx - 1));
                let mut curr = self.buddies[idx - 1];
                let mut prev: Option<u64> = None;

                while let Some(b) = off_to_option(curr) {
                    // sorted by address
                    if b > next {
                        break;
                    }
                    prev = Some(b);
                    curr = self.buddy(b).next;
                }
                // insert the two splitted buddy
                if let Some(p) = prev {
                    self.perform(next, self.buddy(p).next);
                    self.perform(p, next);
                } else {
                    self.perform(next, self.buddies[idx - 1]);
                    self.perform(self.get_off(&self.buddies[idx - 1]), next);
                }
            }
            debug!("return address {res:} and split {split:}");
            Some(res)
        }
    }

    #[inline]
    /// Generates required changes to the meta-data for allocating a new memory
    /// block with the size `len`.
    /// 
    /// If successful, tt returns the offset of the available free block.
    /// otherwise, `u64::MAX` is returned.
    pub unsafe fn alloc_impl(&mut self, len: usize) -> u64 {
        let span = span!(Level::DEBUG, "Buddy alloc");
        let _enter = span.enter();
        debug!("Allocate {len:} bytes");
        let idx = get_idx(len);
        let len = 1 << idx;
        let res;
        if len > self.available {
            res = u64::MAX;
        } else {
            match self.find_free_memory(idx, false) {
                Some(off) => {
                    self.available -= len;
                    res = off + self.base;
                }
                None => {
                    eprintln!(
                        "Cannot find memory slot of size {} (available: {})",
                        len,
                        self.available()
                    );
                    res = u64::MAX;
                }
            }
        }
        res
    }

    #[inline]
    /// Generates required changes to the meta-data for reclaiming the memory
    /// block at offset `off` with the size of `len`.
    pub unsafe fn dealloc_impl(&mut self, off: u64, len: usize) {
        let idx = get_idx(len);
        let len = 1 << idx;

        self.free_impl(off, len);
    }

    #[inline]
    unsafe fn free_impl(&mut self, off: u64, len: usize) {
        let idx = get_idx(len);
        debug!("free {idx:} idx block, {off:} off and {len:} len.");
        let end = off + (1 << idx);
        let mut curr = self.buddies[idx];
        let mut prev: Option<u64> = None;
        if idx < self.last_idx { // Only idx < last_idx, it can be merged.
            while let Some(b) = off_to_option(curr) {
                // get current Buddy, e is the next Buddy addr.
                let e = self.buddy(b);
                let on_left = off & (1 << idx) == 0;
                if (b == end && on_left) || (b + len as u64 == off && !on_left) {
                    let off = off - self.base;
                    let off = off.min(b);
                    if let Some(p) = prev {
                        self.perform(p, e.next);
                    } else {
                        self.perform(self.get_off(&self.buddies[idx]), e.next);
                    }
                    // next recursive will += len << 1
                    self.available -= len;
                    self.free_impl(off+self.base, len << 1);
                    return;
                }
                if b > off {
                    break;
                }
                prev = Some(b);
                curr = e.next;
                debug_assert_ne!(curr, b, "Cyclic link in free_impl");
            }
        }
        let off = off - self.base;
        if let Some(p) = prev {
            self.perform(off, self.buddy(p).next);
            self.perform(p, off);
        } else {
            self.perform(off, self.buddies[idx]);
            self.perform(self.get_off(&self.buddies[idx]), off);
        }
        self.available += len;
    }

    #[inline]
    fn perform(&mut self, off: u64, next: u64) {
        debug!("perform {off:} -> {next:}");
        let n = self.buddy(off);
        n.next = next;
    }

    #[inline]
    /// Determines if the given address range is allocated
    pub fn is_allocated(&mut self, off: u64, _len: usize) -> bool {
        let end = off + _len as u64 - 1;
        let idx = get_idx(_len);
        for idx in idx..self.last_idx + 1 {
            let len = 1 << idx;
            let mut curr = self.buddies[idx];

            while let Some(b) = off_to_option(curr) {
                let r = b + len;
                if (off >= b && off < r) || (end >=b && end < r) || (off <=b && end >=r) {
                   return false; 
                }
                if b > off {
                    break;
                }
                curr = self.buddy(b).next;
                debug_assert_ne!(curr, b, "Cyclic link in is_allocated");
            }
        }
        true
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn available(&self) -> usize {
        self.available
    }

    #[inline]
    pub fn used(&self) -> usize {
        self.size - self.available
    }

}

#[cfg(test)]
mod test {
    use std::{path::PathBuf, str::FromStr};
    use std::vec;
    use tracing;
    use tracing_subscriber;

    use crate::alloc::PmemVBuddyAllocator;

    type VP = PmemVBuddyAllocator;
    const DEFAULT_SIZE: usize = 16 * 1024 * 1024;

    #[test]
    fn buddy_volatile_allocator_test() {
        tracing_subscriber::fmt()
            // enable everything
            .with_max_level(tracing::Level::TRACE)
            // sets this to be the default, global collector for this application.
            .init();
        tracing::debug!("start test");
        let all = VP::new(PathBuf::from_str("/mnt/pmemdir").unwrap(), DEFAULT_SIZE);
        tracing::debug!("initialize an new PmemVBuddyAllocator;");
        let mut vec = vec::Vec::new_in(all);
        tracing::debug!("new a Vec with custom allocator");
        vec.push(0);
        vec.push(1);
        vec.push(2);
        tracing::debug!("filled the data;");
        vec.iter().enumerate()
        .for_each(|(i, v)| {
            assert_eq!(i, *v);
        })
    }
}