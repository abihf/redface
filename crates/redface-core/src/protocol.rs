use rkyv::{Archive, Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum DaemonRequest {
	Authenticate {
		client: String,
		user: String,
		timeout: Option<i32>,
		show_osd: bool,
	},
	Record,
}

#[derive(Clone, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum DaemonResponse {
	AuthSuccess,
	AuthError(String),
	FaceDescriptors(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum OSDNotification {
	Verifying,
	Success,
	FaceMismatch,
	Cancelling,
	Stopped,
}
