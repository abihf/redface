use std::os::unix::net::UnixStream;

use redface_core::{AuthReq, Config, ReadJson, Res};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let config = Config::load_default()?;
	let mut conn = UnixStream::connect(&config.socket)?;
	let uid = unsafe { libc::geteuid() };
	let req = AuthReq {
		client: "check".into(),
		user: uid.to_string(),
		timeout: Some(-1),
		show_osd: true,
	};
	req.write_to(&mut conn)?;
	let res = Res::read_json(&conn)?;
	println!("Result {}", res.status.as_str());
	Ok(())
}
