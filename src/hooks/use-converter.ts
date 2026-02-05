import { useState, useEffect } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import type { ProgressEvent, ConvertResult, Format } from "@/lib/types";

export function useConverter() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [selectedFormat, setSelectedFormat] = useState<Format | null>(null);
  const includeImages = true;
  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!isConverting || !startTime) return;

    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [isConverting, startTime]);

  const selectFile = async () => {
    const file = await open({
      multiple: false,
      filters: [
        { name: "NDJSON", extensions: ["ndjson", "jsonl"] },
        { name: "All Files", extensions: ["*"] },
      ],
    });

    if (file) {
      setSelectedFile(file);
      setSelectedFileName(file.split(/[/\\]/).pop() || file);
      setResult(null);
      setError(null);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setSelectedFileName(null);
    setResult(null);
    setError(null);
    setSelectedFormat(null);
  };

  const setFileFromPath = (path: string) => {
    setSelectedFile(path);
    setSelectedFileName(path.split(/[/\\]/).pop() || path);
    setResult(null);
    setError(null);
  };

  const handleConvert = async () => {
    if (!selectedFile || !selectedFormat) return;

    const basePath = selectedFile.replace(/\.[^.]+$/, "");
    const formatSlug = selectedFormat.name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/(^-|-$)/g, "");

    const outputPath = await save({
      filters: [{ name: "ZIP", extensions: ["zip"] }],
      defaultPath: `${basePath}_${formatSlug}.zip`,
    });

    if (!outputPath) return;

    setIsConverting(true);
    setProgress(null);
    setResult(null);
    setError(null);
    setStartTime(Date.now());
    setElapsedSeconds(0);

    const channel = new Channel<ProgressEvent>();
    channel.onmessage = (event) => {
      setProgress(event);
    };

    try {
      const convertResult = await invoke<ConvertResult>("convert_ndjson", {
        filePath: selectedFile,
        format: selectedFormat.id,
        outputPath,
        includeImages,
        channel,
      });

      setResult(convertResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsConverting(false);
    }
  };

  const resetState = () => {
    setSelectedFile(null);
    setSelectedFileName(null);
    setSelectedFormat(null);
    setResult(null);
    setError(null);
    setProgress(null);
    setStartTime(null);
    setElapsedSeconds(0);
  };

  const getProgressPercentage = () => {
    if (!progress || progress.total === 0) return 0;
    return Math.round((progress.current / progress.total) * 100);
  };

  const formatElapsedTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getDownloadRate = () => {
    if (
      elapsedSeconds === 0 ||
      !progress ||
      progress.phase !== "downloading" ||
      progress.current === 0
    )
      return "0.0";
    return (progress.current / elapsedSeconds).toFixed(1);
  };

  // Expose setFileFromPath for E2E testing (only in dev/test)
  useEffect(() => {
    if (import.meta.env.DEV || import.meta.env.MODE === "test") {
      (window as unknown as { __E2E_SET_FILE__?: (path: string) => void }).__E2E_SET_FILE__ = setFileFromPath;
    }
    return () => {
      delete (window as unknown as { __E2E_SET_FILE__?: unknown }).__E2E_SET_FILE__;
    };
  }, []);

  return {
    selectedFile,
    selectedFileName,
    selectedFormat,
    setSelectedFormat,
    isConverting,
    progress,
    result,
    error,
    elapsedSeconds,
    selectFile,
    removeFile,
    setFileFromPath,
    handleConvert,
    resetState,
    getProgressPercentage,
    formatElapsedTime,
    getDownloadRate,
  };
}
