import React, { useState } from "react";

import {
  CheckCircle,
  DatabaseZap,
  Download,
  Loader,
  Play,
  XCircle,
} from "lucide-react";
import { Cohort, PhenotypeSummary } from "@/app/lib/types";
import {
  API_URL,
  PostGWASResponse,
  PvaluesResult,
  runGWAS,
} from "@/app/lib/api";
import { convertToRPN, isSingleField, Phenotype } from "@/lib/phenotype";
import { downloadPvals, pollJobStatus, summarize } from "@/lib/gwas";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";

export function GwasHandlerNoValidate({
  cohort,
  phenotype,
  setIsMutable,
  setFitQuality,
  setResults,
  handleReset,
}: {
  cohort: Cohort;
  phenotype: Phenotype;
  setIsMutable: React.Dispatch<React.SetStateAction<boolean>>;
  setFitQuality: React.Dispatch<React.SetStateAction<PhenotypeSummary | null>>;
  setResults: React.Dispatch<React.SetStateAction<PvaluesResult | null>>;
  handleReset: React.MouseEventHandler;
}) {
  const [status, setStatus] = useState<
    "running" | "done" | "error" | "cached" | null
  >(null);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const phenotypeString = convertToRPN(phenotype);
  const isTrivialPhenotype = isSingleField(phenotype);

  const handleRunGWAS: React.MouseEventHandler<
    HTMLButtonElement
  > = async () => {
    setIsMutable(false);
    setStatus("running");
    if (!isTrivialPhenotype) {
      const result = await summarize(cohort, phenotype);
      if (result.ok) {
        setFitQuality(result.fitQuality);
      } else {
        setError(result.error);
        return;
      }
    }
    let gwasResponse: PostGWASResponse;
    try {
      gwasResponse = await runGWAS(phenotypeString, cohort);
      setRequestId(gwasResponse.request_id);
      if (gwasResponse.status === "cached" || gwasResponse.status === "done") {
        setStatus(gwasResponse.status);
      } else if (gwasResponse.status === "error") {
        throw new Error(gwasResponse.message || "Unknown error");
      } else {
        const result = await pollJobStatus(gwasResponse.request_id);
        setStatus(result.status);
        setError(result.error);
      }
    } catch (err) {
      let errorMessage = "Error running GWAS";
      if (err instanceof Error) {
        errorMessage = `Error running GWAS: ${err.message}`;
      }
      console.log(errorMessage);
      setError(errorMessage);
      return;
    }
    const result = await downloadPvals(
      gwasResponse.request_id,
      cohort,
      phenotype,
    );
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setResults(result.result);
  };

  return (
    <div className="flex flex-wrap gap-4 justify-between">
      <div className="flex flex-wrap gap-4">
        {status === null && <RunGwasButton handleRunGWAS={handleRunGWAS} />}
        {status !== null && <JobStatusBoxes status={status} error={error} />}
        {requestId && (status === "done" || status === "cached") && (
          <DownloadButton requestId={requestId} />
        )}
      </div>
      <div className="flex flex-wrap">
        <ResetButton
          handleReset={handleReset}
          isMutable={status !== "running"}
        />
      </div>
    </div>
  );
}

function RunGwasButton({
  handleRunGWAS,
}: {
  handleRunGWAS: React.MouseEventHandler<HTMLButtonElement>;
}) {
  return (
    <button
      onClick={handleRunGWAS}
      className="bg-green-main hover:bg-green-dark text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors"
    >
      <Play className="mr-2" size={20} />
      Run GWAS
    </button>
  );
}

function JobStatusBoxes({
  status,
  error,
}: {
  status: "running" | "done" | "error" | "cached";
  error: string | null;
}) {
  function getIcon(status: "running" | "done" | "error" | "cached") {
    switch (status) {
      case "running":
        return <Loader className="animate-spin mr-2" size={20} />;
      case "done":
        return <CheckCircle className="mr-2 text-green-main" size={20} />;
      case "error":
        return <XCircle className="mr-2 text-red-600" size={20} />;
      case "cached":
        return <DatabaseZap className="mr-2 text-green-main" size={20} />;
    }
  }

  function getText(status: "running" | "done" | "error" | "cached") {
    switch (status) {
      case "running":
        return "Job running...";
      case "done":
        return "GWAS completed.";
      case "error":
        if (error === null) {
          return "GWAS failed. Please try again.";
        } else {
          return "GWAS failed. Please try again. Error message: " + error;
        }
      case "cached":
        return "GWAS cached.";
    }
  }

  return (
    <div className="flex items-center bg-gray-100 p-2 rounded-lg">
      {getIcon(status)}
      <span className="text-gray-700">{getText(status)}</span>
    </div>
  );
}

function DownloadButton({ requestId }: { requestId: string }) {
  return (
    <div>
      <a href={`${API_URL}/download/${requestId}`} download>
        <button className="bg-primary hover:bg-blue-dark text-primary-foreground text-l font-semibold py-1 px-4 h-[38px] rounded-lg flex gap-1 items-center transition-colors text-nowrap">
          <Download className="mr-2" size={20} />
          Download Results
        </button>
      </a>
    </div>
  );
}

function ResetButton({
  handleReset,
  isMutable,
}: {
  handleReset: React.MouseEventHandler;
  isMutable: boolean;
}) {
  const button = (
    <button
      onClick={handleReset}
      disabled={!isMutable}
      className={`bg-red-500 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors ${
        !isMutable ? "cursor-not-allowed" : ""
      }`}
    >
      <XCircle className="mr-2 text-white" size={20} />
      Reset analysis
    </button>
  );

  if (isMutable) {
    return button;
  }
  return (
    <TooltipProvider delayDuration={0} skipDelayDuration={0}>
      <Tooltip>
        <TooltipTrigger>{button}</TooltipTrigger>
        <TooltipContent className="bg-background text-foreground border">
          {!isMutable &&
            "Analysis cannot be reset while running. Please wait for the current job to finish."}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
