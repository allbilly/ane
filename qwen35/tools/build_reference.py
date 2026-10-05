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
    args = parser.parse_args()
    if not args.cargo:
        parser.error("cargo is required for the optional independent reference")
    if not args.uzu.exists():
        subprocess.run(["git", "clone", "https://github.com/trymirai/uzu", str(args.uzu)], check=True)
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
    examples = engine / "examples"
    examples.mkdir(exist_ok=True)
    (examples / "ane_reference.rs").write_text('''use std::path::Path;
use uzu_engine::{backends::cpu::Cpu, engine::Engine};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let engine = Engine::<Cpu>::new()?;
    let model = engine.load_language_model(Path::new(&args[1]))?;
    let tokens: Vec<u32> = args[2].split(',').map(|s| s.parse().unwrap()).collect();
    model.ane_reference(&tokens, Path::new(&args[3]))
}
''')
    subprocess.run([args.cargo, "build", "--release", "-p", "uzu-engine", "--no-default-features",
                    "--features", "cpu", "--example", "ane_reference", "-j", "4"], cwd=args.uzu, check=True)


if __name__ == "__main__":
    main()
