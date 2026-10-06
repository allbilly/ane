"""Build an independent Uzu CPU oracle; never alter the user's ~/uzu checkout."""
import argparse
import shutil
import subprocess
from pathlib import Path

REVISION = "4d880766f5a4137b63d56c3924e7a2e403b0ccc6"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uzu", type=Path, default=Path.home() / ".cache/ane-qwen35/uzu")
    parser.add_argument("--cargo", default=shutil.which("cargo"))
    parser.add_argument("--lockfile", type=Path, help="Reuse a captured Cargo lock for cross-host comparison")
    args = parser.parse_args()
    if not args.cargo:
        parser.error("cargo is required for the optional independent reference")
    if not args.uzu.exists():
        # Fetch the recorded revision, rather than cloning today's HEAD and
        # immediately failing the revision check on a new machine.
        subprocess.run(["git", "init", str(args.uzu)], check=True)
        subprocess.run(["git", "-C", str(args.uzu), "remote", "add", "origin",
                        "https://github.com/trymirai/uzu"], check=True)
        subprocess.run(["git", "-C", str(args.uzu), "fetch", "--depth", "1", "origin", REVISION], check=True)
        subprocess.run(["git", "-C", str(args.uzu), "checkout", "--detach", "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.uzu, text=True).strip()
    if revision != REVISION:
        raise ValueError(f"reference requires Uzu {REVISION}, found {revision}")
    engine = args.uzu / "crates/uzu-engine"
    language = engine / "src/engine/language_model"
    shutil.copyfile(Path(__file__).with_name("uzu_reference.rs"), language / "ane_reference.rs")
    module = language / "mod.rs"
    content = module.read_text()
    if "pub mod ane_reference;" not in content:
        module.write_text(content + "\npub mod ane_reference;\n")
    # A separate crate avoids the engine's large, unrelated dev dependencies.
    cli = args.uzu.parent / "uzu-reference-cli"
    (cli / "src").mkdir(parents=True, exist_ok=True)
    (cli / "Cargo.toml").write_text('''[package]
name = "uzu-reference-cli"
version = "0.1.0"
edition = "2024"
[workspace]
[dependencies]
uzu-engine = {path = "''' + str(engine.resolve()) + '''", default-features = false, features = ["cpu"]}
''')
    (cli / "src/main.rs").write_text('''use std::path::Path;
use uzu_engine::{backends::cpu::Cpu, engine::Engine};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let engine = Engine::<Cpu>::new()?;
    let model = engine.load_language_model(Path::new(&args[1]))?;
    let tokens: Vec<u32> = args[2].split(',').map(|s| s.parse().unwrap()).collect();
    model.ane_reference(&tokens, Path::new(&args[3]))
}
''')
    command = [args.cargo, "build", "--release", "-j", "4"]
    if args.lockfile:
        if args.lockfile.resolve() != (cli / "Cargo.lock").resolve():
            shutil.copyfile(args.lockfile, cli / "Cargo.lock")
        command.append("--locked")
    subprocess.run(command, cwd=cli, check=True)


if __name__ == "__main__":
    main()
