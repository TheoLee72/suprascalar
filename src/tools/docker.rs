use super::Tool;
use crate::error::{Result, SuprascalarError};
use bollard::Docker;
use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
use bollard::models::ContainerCreateBody;
use bollard::query_parameters::{
    CreateContainerOptions, CreateImageOptions, KillContainerOptions, StartContainerOptions,
    StopContainerOptionsBuilder,
};
use bollard::service::HostConfig;
use futures_util::StreamExt;
use regex::Regex;
use serde_json::{Value, json};
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use tokio::runtime::Runtime;
use tokio::time::{Duration, timeout};

/// Docker 샌드박스 도구 (Optimized)
pub struct DockerShell {
    runtime: Runtime,
    docker: Docker,
    container_id: String,
    cwd: Mutex<PathBuf>,
    // 안전 장치 활성화 플래그
    safety_enabled: bool,
}

impl DockerShell {
    pub fn new() -> Result<Self> {
        let runtime = Runtime::new().map_err(|e| SuprascalarError::Unknown(e.to_string()))?;

        // 1. Docker 데몬 연결
        let docker = Docker::connect_with_local_defaults()
            .map_err(|e| SuprascalarError::Unknown(format!("Docker connect failed: {}", e)))?;

        // 2. 호스트 경로 바인딩 준비
        let host_cwd = env::current_dir().map_err(SuprascalarError::Io)?;
        let host_cwd_str = host_cwd.to_string_lossy().to_string();
        let mount_config = format!("{}:/workspace", host_cwd_str);

        // [최적화 1] 가벼운 이미지 사용 (Debian Slim)
        let image_name = "debian:bullseye-slim";

        // 3. 컨테이너 설정
        let config = ContainerCreateBody {
            image: Some(String::from(image_name)),
            cmd: Some(vec![String::from("sleep"), String::from("infinity")]),
            working_dir: Some(String::from("/workspace")),
            host_config: Some(HostConfig {
                memory: Some(512 * 1024 * 1024), // 512MB
                cpu_shares: Some(512),
                // 프로세스 수 제한: 포크밤 등으로 인한 PID 고갈을 방지
                pids_limit: Some(128),
                binds: Some(vec![mount_config]),
                auto_remove: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };

        // 4. 컨테이너 생성 및 실행 (이미지가 없으면 자동 pull)
        let container_id = runtime
            .block_on(async {
                // 이미지 존재 여부 체크 후 필요 시 pull
                if let Err(_) = docker.inspect_image(image_name).await {
                    eprintln!(
                        ">> [Docker] Image '{}' not found locally. Pulling...",
                        image_name
                    );
                    let mut stream = docker.create_image(
                        Some(CreateImageOptions {
                            from_image: Some(String::from(image_name)),
                            ..Default::default()
                        }),
                        None,
                        None,
                    );

                    while let Some(progress) = stream.next().await {
                        match progress {
                            Ok(status) => {
                                if let Some(detail) = status.status {
                                    println!(">> [Docker] {}", detail);
                                }
                            }
                            Err(e) => {
                                return Err(bollard::errors::Error::IOError {
                                    err: std::io::Error::new(
                                        std::io::ErrorKind::Other,
                                        format!("Image pull failed: {}", e),
                                    ),
                                });
                            }
                        }
                    }
                }

                let id = docker
                    .create_container(None::<CreateContainerOptions>, config)
                    .await?
                    .id;

                docker
                    .start_container(&id, None::<StartContainerOptions>)
                    .await?;
                Ok::<String, bollard::errors::Error>(id)
            })
            .map_err(|e| {
                SuprascalarError::Unknown(format!(
                    "Failed to start Docker sandbox: {}. (Try 'docker pull {}')",
                    e, image_name
                ))
            })?;

        println!(
            ">> [Docker] Sandbox Ready (Limit: 512MB). ID: {:.8}",
            container_id
        );

        Ok(Self {
            runtime,
            docker,
            container_id,
            cwd: Mutex::new(PathBuf::from("/workspace")),
            safety_enabled: true,
        })
    }

    /// [Safety 1] 위험한 명령어 차단
    fn check_safety(&self, cmd: &str) -> Result<()> {
        if !self.safety_enabled {
            return Ok(());
        }

        let dangerous_patterns = [
            (
                r"(^|\s)vim?(\s|$)",
                "Interactive editors (vim) hang the agent.",
            ),
            (
                r"(^|\s)nano(\s|$)",
                "Interactive editors (nano) hang the agent.",
            ),
            (r":\(\)\{\s*:\|:&", "Fork bombs are forbidden."),
            // workspace 내부 전체 삭제 방지
            (
                r"rm\s+-[rRf]+\s+(/workspace|\.|/)$",
                "Mass deletion of workspace is forbidden.",
            ),
        ];

        for (pattern, reason) in dangerous_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(cmd) {
                    return Err(SuprascalarError::Unknown(format!(
                        "SECURITY BLOCK: Command '{}' blocked. Reason: {}",
                        cmd, reason
                    )));
                }
            }
        }
        Ok(())
    }

    /// [Safety 2] 호스트 Git 스냅샷 생성
    /// Docker 내부가 아닌 '호스트'에서 git 명령을 실행합니다.
    fn create_git_snapshot(&self, cmd_context: &str) {
        if !self.safety_enabled {
            return;
        }

        // 호스트의 현재 작업 디렉토리 (여기가 /workspace로 마운트되어 있음)
        let host_cwd = match env::current_dir() {
            Ok(p) => p,
            Err(_) => return,
        };

        // .git 확인
        if !host_cwd.join(".git").exists() {
            return;
        }

        // 변경사항 확인
        let status = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&host_cwd)
            .output();

        if let Ok(output) = status {
            if !output.status.success() || output.stdout.is_empty() {
                return; // 변경 없음
            }

            // Auto-Commit
            let _ = Command::new("git")
                .args(["add", "."])
                .current_dir(&host_cwd)
                .output();

            let msg = format!(
                "Suprascalar Auto-save: Before running '{}' in Docker",
                cmd_context
            );
            let _ = Command::new("git")
                .args(["commit", "-m", &msg])
                .current_dir(&host_cwd)
                .output();
        }
    }
}

// 프로그램 종료 시 컨테이너 정리 (Cleanup)
impl Drop for DockerShell {
    fn drop(&mut self) {
        let container_id = self.container_id.clone();
        println!(">> [Docker] Graceful shutdown initiated (Timeout: 3s)...");
        let _ = self.runtime.block_on(async {
            let options = StopContainerOptionsBuilder::new().t(3).build();
            match self
                .docker
                .stop_container(&container_id, Some(options))
                .await
            {
                Ok(_) => println!(">> [Docker] Container stopped gracefully."),
                Err(e) => {
                    eprintln!(">> [Docker] Stop failed ({}). Forcing kill...", e);
                    let _ = self
                        .docker
                        .kill_container(&container_id, None::<KillContainerOptions>)
                        .await;
                }
            }
        });
    }
}

impl Tool for DockerShell {
    fn name(&self) -> &str {
        "run_shell_command"
    }

    fn description(&self) -> &str {
        "Executes shell commands in a Docker sandbox with persistent state. \
         Modifications to /workspace are reflected on the host. \
         Auto-commits to Git before execution for safety."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    fn execute(&self, args: Value) -> Result<String> {
        let command_str = args["command"]
            .as_str()
            .ok_or_else(|| SuprascalarError::Unknown("Missing 'command' parameter".to_string()))?;

        // 1. [Safety] 금지어 검사
        self.check_safety(command_str)?;

        // 2. [Safety] Git 스냅샷 (호스트에서 실행)
        // 파일 수정, 이동, 삭제 등이 포함될 수 있으므로 일단 모든 명령 전에 체크
        self.create_git_snapshot(command_str);

        // 3. 현재 Docker 내부 경로 가져오기
        let current_cwd = self.cwd.lock().unwrap().to_string_lossy().to_string();

        // 4. 명령어 주입 (Marker 전략)
        let marker = "___SUPRA_CWD";
        let injected_command = format!("{}; echo \"{}:$(pwd)\"", command_str, marker);

        let timeout_duration = Duration::from_secs(60);

        // 5. Docker Exec 실행
        let output_result = self.runtime.block_on(async {
            let execution_future = async {
                let exec_config = CreateExecOptions {
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    working_dir: Some(current_cwd.as_str()),
                    cmd: Some(vec!["/bin/sh", "-c", &injected_command]),
                    ..Default::default()
                };

                let exec_id = self
                    .docker
                    .create_exec(&self.container_id, exec_config)
                    .await?
                    .id;

                // Stream Start
                let mut combined_output = String::new();
                let stream = self
                    .docker
                    .start_exec(&exec_id, None::<StartExecOptions>)
                    .await?;

                match stream {
                    StartExecResults::Attached { mut output, .. } => {
                        while let Some(msg) = output.next().await {
                            if let Ok(log) = msg {
                                // [UX] 실시간 터미널 출력
                                print!("{}", log);
                                combined_output.push_str(&log.to_string());
                            }
                        }
                    }
                    StartExecResults::Detached => {
                        combined_output.push_str("Exec started in detached mode")
                    }
                }
                Ok::<String, bollard::errors::Error>(combined_output)
            };
            // [Time Limit] 비동기 작업에 타임아웃 걸기
            match timeout(timeout_duration, execution_future).await {
                Ok(result) => result, // 시간 내 완료됨
                Err(_) => {
                    // 시간 초과 발생!
                    // 여기서 컨테이너 전체를 죽일 필요는 없고, 그냥 에러 메시지만 반환하면
                    // 다음 턴에서 에이전트가 "아, 너무 오래 걸려서 실패했구나"라고 인지함.
                    // (필요하다면 여기서 exec process를 kill 하는 로직을 추가할 수도 있음)
                    Ok(format!(
                        "Error: Command timed out after {} seconds.",
                        timeout_duration.as_secs()
                    ))
                }
            }
        });

        match output_result {
            Ok(full_output) => {
                // 6. 결과 파싱 및 상태 업데이트
                let mut lines: Vec<&str> = full_output.lines().collect();
                let mut final_output = full_output.clone();
                let mut new_cwd_found = false;

                if let Some(last_line) = lines.last() {
                    if last_line.contains(marker) {
                        if let Some(path_str) = last_line.strip_prefix(&format!("{}:", marker)) {
                            let new_path = PathBuf::from(path_str.trim());
                            *self.cwd.lock().unwrap() = new_path;
                            new_cwd_found = true;
                        }
                    }
                }

                if new_cwd_found {
                    lines.pop(); // 마커 라인 제거
                    final_output = lines.join("\n");
                }

                // 7. 출력 제한
                if final_output.len() > 2000 {
                    Ok(format!("{}\n... [Truncated] ...", &final_output[..2000]))
                } else if final_output.trim().is_empty() {
                    Ok("(Command executed successfully)".to_string())
                } else {
                    Ok(final_output)
                }
            }
            Err(e) => Ok(format!("Docker Execution Failed: {}", e)),
        }
    }
}
