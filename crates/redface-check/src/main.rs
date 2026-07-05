use std::os::unix::net::UnixStream;

use redface_runtime::{Config, read_res, write_auth_req};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::load_default()?;
    let mut conn = UnixStream::connect(&config.socket)?;
    let uid = unsafe { libc::geteuid() };
    write_auth_req(&mut conn, &uid.to_string(), "check")?;
    let res = read_res(&mut conn)?;
    println!("Result {}", res.status.as_str());
    Ok(())
}
