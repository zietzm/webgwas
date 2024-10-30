import { Cohort, PhenotypeSummary } from "@/app/lib/types";
import {
  getPhenotypeSummary,
  getPvalues,
  getResults,
  PvaluesResult,
} from "@/app/lib/api";
import { Phenotype, convertToRPN, getCodes } from "@/lib/phenotype";

export interface downloadPvalsResult {
  ok: boolean;
  error: string | null;
  result: PvaluesResult | null;
}

export async function downloadPvals(
  requestId: string,
  cohort: Cohort,
  phenotype: Phenotype,
): Promise<downloadPvalsResult> {
  try {
    const featureCodes = getCodes(phenotype);
    const result = await getPvalues(requestId, cohort.id, featureCodes);
    return {
      ok: true,
      error: null,
      result: result,
    };
  } catch (err) {
    let errorMessage = "Error downloading pvalues";
    if (err instanceof Error) {
      errorMessage = `Error downloading pvalues: ${err.message}`;
    }
    return {
      ok: false,
      error: errorMessage,
      result: null,
    };
  }
}

export interface summarizeResult {
  ok: boolean;
  fitQuality: PhenotypeSummary | null;
  error: string | null;
}

export async function summarize(
  cohort: Cohort,
  phenotype: Phenotype,
): Promise<summarizeResult> {
  const phenotypeDefinition = convertToRPN(phenotype);
  try {
    const phenotypeSummary = await getPhenotypeSummary(
      phenotypeDefinition,
      cohort,
    );
    return {
      ok: true,
      fitQuality: phenotypeSummary,
      error: null,
    };
  } catch (err) {
    let errorMessage = "Error validating the phenotype";
    if (err instanceof Error) {
      errorMessage = `Error validating the phenotype: ${err.message}`;
    }
    return {
      ok: false,
      fitQuality: null,
      error: errorMessage,
    };
  }
}

export interface PollResult {
  status: "running" | "done" | "error" | "cached" | null;
  error: string | null;
}

export async function pollJobStatus(requestId: string): Promise<PollResult> {
  return new Promise((resolve) => {
    const checkStatus = async () => {
      try {
        const result = await getResults(requestId);

        switch (result.status) {
          case "cached":
            resolve({ status: "cached", error: null });
            break;
          case "done":
            resolve({ status: "done", error: null });
            break;
          case "error":
            resolve({ status: "error", error: result.error_msg });
            break;
          default:
            // Schedule next check and continue polling
            setTimeout(checkStatus, 1000); // Poll every second
        }
      } catch (err) {
        resolve({ status: "error", error: "Unknown error" });
      }
    };

    // Start the polling
    checkStatus();
  });
}
