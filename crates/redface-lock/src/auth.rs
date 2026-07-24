use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::io;
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::thread::{self, JoinHandle};

use redface_core::{AuthReq, ReadJson, Res, Status};

pub const PAM_SERVICE: &str = "redface-lock";

pub enum AuthEvent {
	Pam(Result<(), String>),
	Face(Result<(), String>),
}

// ---------------------------------------------------------------------------
// Minimal client-side PAM FFI (pam-client/pam-sys bindgen against libclang,
// which conflicts with this workspace's clang runtime feature; the surface we
// need is four functions and one callback).
// ---------------------------------------------------------------------------

const PAM_SUCCESS: c_int = 0;
const PAM_PROMPT_ECHO_OFF: c_int = 1;
const PAM_PROMPT_ECHO_ON: c_int = 2;
const PAM_AUTH_ERR: c_int = 7;

enum PamHandle {}

#[repr(C)]
struct PamMessage {
	msg_style: c_int,
	msg: *const c_char,
}

#[repr(C)]
struct PamResponse {
	resp: *mut c_char,
	resp_retcode: c_int,
}

#[repr(C)]
struct PamConv {
	conv: extern "C" fn(c_int, *mut *mut PamMessage, *mut *mut PamResponse, *mut c_void) -> c_int,
	appdata_ptr: *mut c_void,
}

#[link(name = "pam")]
unsafe extern "C" {
	fn pam_start(
		service: *const c_char,
		user: *const c_char,
		conv: *const PamConv,
		handle: *mut *mut PamHandle,
	) -> c_int;
	fn pam_authenticate(handle: *mut PamHandle, flags: c_int) -> c_int;
	fn pam_acct_mgmt(handle: *mut PamHandle, flags: c_int) -> c_int;
	fn pam_end(handle: *mut PamHandle, status: c_int) -> c_int;
}

struct Credentials {
	username: CString,
	password: CString,
}

/// Answers every echo-off prompt with the entered password and every echo-on
/// prompt (e.g. a username re-request) with the username. Must not panic: it
/// is called from libpam.
extern "C" fn converse(
	num_msg: c_int,
	messages: *mut *mut PamMessage,
	out_responses: *mut *mut PamResponse,
	appdata: *mut c_void,
) -> c_int {
	if messages.is_null() || out_responses.is_null() || appdata.is_null() || num_msg < 0 {
		return PAM_AUTH_ERR;
	}
	unsafe {
		let credentials = &*(appdata as *const Credentials);
		let responses = libc::calloc(num_msg as usize, std::mem::size_of::<PamResponse>()) as *mut PamResponse;
		if responses.is_null() {
			return PAM_AUTH_ERR;
		}
		for i in 0..num_msg as isize {
			let message = *messages.offset(i);
			let answer = match (*message).msg_style {
				PAM_PROMPT_ECHO_OFF => libc::strdup(credentials.password.as_ptr()),
				PAM_PROMPT_ECHO_ON => libc::strdup(credentials.username.as_ptr()),
				// Info/error messages expect no text back.
				_ => std::ptr::null_mut(),
			};
			(*responses.offset(i)).resp = answer;
			(*responses.offset(i)).resp_retcode = 0;
		}
		*out_responses = responses;
	}
	PAM_SUCCESS
}

/// Validates a password against the "redface-lock" PAM service.
pub fn pam_auth(username: &str, password: &str) -> Result<(), String> {
	let credentials = Credentials {
		username: CString::new(username).map_err(|_| "username contains NUL".to_owned())?,
		password: CString::new(password).map_err(|_| "password contains NUL".to_owned())?,
	};
	let service = CString::new(PAM_SERVICE).expect("static service name has no NUL");
	let conv = PamConv {
		conv: converse,
		appdata_ptr: &credentials as *const Credentials as *mut c_void,
	};

	let mut handle: *mut PamHandle = std::ptr::null_mut();
	let mut status = unsafe { pam_start(service.as_ptr(), credentials.username.as_ptr(), &conv, &mut handle) };
	if status == PAM_SUCCESS {
		status = unsafe { pam_authenticate(handle, 0) };
		if status == PAM_SUCCESS {
			status = unsafe { pam_acct_mgmt(handle, 0) };
		}
		unsafe { pam_end(handle, status) };
	}
	match status {
		PAM_SUCCESS => Ok(()),
		PAM_AUTH_ERR => Err("wrong password".to_owned()),
		code => Err(format!("pam error {code}")),
	}
}

/// The username of the process owner (used for both PAM and the face model lookup).
pub fn current_username() -> io::Result<String> {
	unsafe {
		let pw = libc::getpwuid(libc::geteuid());
		if pw.is_null() {
			return Err(io::Error::other("getpwuid failed"));
		}
		Ok(CStr::from_ptr((*pw).pw_name).to_string_lossy().into_owned())
	}
}

/// A running face-recognition request against the redfaced daemon.
///
/// Dropping the connection cancels the verification server-side (the daemon's
/// disconnect watcher stops the camera stream), so `stop()` is just a shutdown.
pub struct FaceAuth {
	cancel: UnixStream,
	cancelled: Arc<AtomicBool>,
	join: JoinHandle<()>,
}

impl FaceAuth {
	pub fn start(socket_path: &str, uid: u32, tx: Sender<AuthEvent>, wake: UnixStream) -> io::Result<Self> {
		use std::io::Write;
		let mut conn = UnixStream::connect(socket_path)?;
		let cancel = conn.try_clone()?;
		let cancelled = Arc::new(AtomicBool::new(false));
		let thread_cancelled = cancelled.clone();
		let join = thread::spawn(move || {
			let result = run(&mut conn, uid);
			// A user-initiated stop races with the daemon's reply; don't report it.
			if !thread_cancelled.load(Ordering::Relaxed) {
				let _ = tx.send(AuthEvent::Face(result));
				let mut wake = wake;
				let _ = wake.write_all(&[1]);
			}
		});
		Ok(Self {
			cancel,
			cancelled,
			join,
		})
	}

	pub fn stop(self) {
		let FaceAuth {
			cancel,
			cancelled,
			join,
		} = self;
		cancelled.store(true, Ordering::Relaxed);
		let _ = cancel.shutdown(Shutdown::Both);
		let _ = join.join();
	}
}

fn run(conn: &mut UnixStream, uid: u32) -> Result<(), String> {
	AuthReq {
		client: "lock".into(),
		user: uid.to_string(),
		..Default::default()
	}
	.write_to(&mut *conn)
	.map_err(|e| format!("daemon request failed: {e}"))?;
	let res = Res::read_json(&mut *conn).map_err(|e| format!("daemon response failed: {e}"))?;
	match res.status {
		Status::Success => Ok(()),
		Status::Error => Err(if res.error.is_empty() {
			"face not recognized".to_owned()
		} else {
			res.error
		}),
	}
}
