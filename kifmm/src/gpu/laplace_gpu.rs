unsafe extern  "C" {
    fn add_kernel(a: *mut f32, b: *mut f32, c: *mut f32, n: i32);
}