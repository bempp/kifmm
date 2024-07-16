#[no_mangle]
pub extern "C" fn add_from_rust(left: usize, right: usize) -> usize {
    left + right
}

#[no_mangle]
pub extern "C" fn hello_world() {
    println!("Hello world")
}
