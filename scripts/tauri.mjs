import { spawnSync } from "node:child_process";

const args = process.argv.slice(2);
const command = args[0] ?? "";
const extraArgs = args.slice(1);

const devConfig = ["-c", "src-tauri/tauri.dev.conf.json"];

const finalArgs =
  command === "dev" ? ["dev", ...devConfig, ...extraArgs] : [command, ...extraArgs];

const result = spawnSync("tauri", finalArgs, { stdio: "inherit" });
process.exit(result.status ?? 1);
