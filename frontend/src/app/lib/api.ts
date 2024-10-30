import {
  Cohort,
  PhenotypeNode,
  isFeature,
  isOperator,
  isConstant,
  Feature,
  PhenotypeSummary,
  ListNode,
} from "./types";

export const API_URL: string = process.env.NEXT_PUBLIC_API_URL || "";

export interface PostGWASResponse {
  request_id: string;
  status: "queued" | "done" | "error" | "cached";
  message: string | null;
}

export interface ValidationResponse {
  is_valid: boolean;
  message: string;
}

export interface ResultsResponse {
  request_id: string;
  status: "queued" | "done" | "error" | "cached";
  error_msg: string | null;
}

export interface Pvalue {
  index: number;
  pvalue: number;
  chromosome: string;
  color: number;
  label: string;
}

export interface ChromosomePosition {
  chromosome: string;
  midpoint: number;
}

export interface PvaluesResponse {
  request_id: string;
  status: "queued" | "cached" | "done" | "error";
  error_msg?: string;
  pvalues?: Pvalue[];
  chromosome_positions?: ChromosomePosition[];
  color_map?: string[];
}

export interface PvaluesResult {
  pvalues: Pvalue[];
  chromosome_positions: ChromosomePosition[];
  color_map: string[];
}

export async function fetchCohorts(): Promise<Cohort[]> {
  const response = await fetch(`${API_URL}/cohorts`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    throw new Error("Failed to fetch cohorts");
  }
  const data = await response.json();
  return data as Cohort[];
}

export async function fetchFeatures(cohort: Cohort): Promise<Feature[]> {
  const myUrl = new URL(`${API_URL}/features`);
  myUrl.searchParams.set("cohort_id", cohort.id.toString());
  const response = await fetch(myUrl.href, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      "Accept-Encoding": "zstd",
    },
  });
  if (!response.ok) {
    throw new Error("Failed to fetch nodes");
  }
  const data = await response.json();
  const returnData: Feature[] = data.map((d: any) => {
    return {
      code: d.c,
      name: d.n,
      type: d.t,
      sample_size: d.s,
    };
  });
  return returnData;
}

export async function validatePhenotype(
  phenotypeDefinition: string,
  cohort: Cohort,
): Promise<ValidationResponse> {
  const myUrl = new URL(`${API_URL}/phenotype`);
  const response = await fetch(myUrl.href, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      phenotype_definition: phenotypeDefinition,
      cohort_id: cohort.id,
    }),
  });
  if (response.status === 400) {
    const text = await response.text();
    console.log(text);
  }
  if (!response.ok) {
    throw new Error("Failed to validate phenotype");
  }
  const result = await response.json();
  return result as ValidationResponse;
}

export async function getPhenotypeSummary(
  phenotypeDefinition: string,
  selectedCohort: Cohort,
): Promise<PhenotypeSummary> {
  const myUrl = new URL(`${API_URL}/phenotype_summary`);
  const response = await fetch(myUrl.href, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept-Encoding": "zstd",
    },
    body: JSON.stringify({
      phenotype_definition: phenotypeDefinition,
      cohort_id: selectedCohort!.id,
      n_samples: 2000,
    }),
  });
  if (!response.ok) {
    throw new Error("Failed to get phenotype summary");
  }
  const result = await response.json();
  return result as PhenotypeSummary;
}

export async function runGWAS(
  phenotypeDefinition: string,
  selectedCohort: Cohort,
): Promise<PostGWASResponse> {
  const myUrl = new URL(`${API_URL}/igwas`);
  const response = await fetch(myUrl.href, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      phenotype_definition: phenotypeDefinition,
      cohort_id: selectedCohort!.id,
    }),
  });
  if (!response.ok) {
    throw new Error("Failed to run GWAS");
  }
  const result = await response.json();
  return result as PostGWASResponse;
}

export async function getResults(requestId: string): Promise<ResultsResponse> {
  const myUrl = new URL(`${API_URL}/igwas/results/${requestId}`);
  const response = await fetch(myUrl.href, {
    method: "GET",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch results: ${response.status}`);
  }
  const result = await response.json();
  return result as ResultsResponse;
}

export async function getPvalues(
  requestId: string,
  cohortId: number,
  feature_codes: string[],
): Promise<PvaluesResult> {
  const myUrl = new URL(`${API_URL}/igwas/results/pvalues/${requestId}`);
  const response = await fetch(myUrl.href, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "Accept-Encoding": "zstd",
    },
    body: JSON.stringify({
      cohort_id: cohortId,
      minp: 1.3, // Hard coded to -log10(0.05) for faster rendering
      features: feature_codes,
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch results: ${response.status}`);
  }
  const rawResult: any = await response.json();
  if (
    rawResult.pvalues !== null &&
    rawResult.chromosome_positions !== null &&
    rawResult.color_map !== null
  ) {
    const formattedPvalues: Pvalue[] = rawResult.pvalues.map((p: any) => {
      return {
        index: p.i,
        pvalue: p.p,
        chromosome: p.c,
        color: p.g,
        label: p.l,
      };
    });
    const formattedChromosomePositions: ChromosomePosition[] =
      rawResult.chromosome_positions.map((p: any) => {
        return {
          chromosome: p.c,
          midpoint: p.m,
        };
      });

    const colorMap: string[] = rawResult.color_map;
    return {
      pvalues: formattedPvalues,
      chromosome_positions: formattedChromosomePositions,
      color_map: colorMap,
    };
  }
  console.log(rawResult);
  throw new Error("Failed to get pvalues");
}

export function convertFeaturetoRPN(node: Feature): string {
  return '"' + node.code + '"';
}

export function convertNodeToRPN(node: PhenotypeNode): string {
  if (isFeature(node.data)) {
    return convertFeaturetoRPN(node.data);
  } else if (isOperator(node.data)) {
    if (node.data.name === "Root") {
      return "";
    } else {
      return "`" + node.data.name + "`";
    }
  } else if (isConstant(node.data)) {
    return `<REAL:${node.data.value}>`;
  } else {
    throw new Error("Invalid node type" + node.data);
  }
}

export function convertTreeToRPN(node: PhenotypeNode): string {
  const nodeString = convertNodeToRPN(node);
  if (isFeature(node.data) || isConstant(node.data)) {
    return nodeString;
  } else if (isOperator(node.data)) {
    const childrenRPN = node.children.map(convertTreeToRPN);
    if (node.data.name === "Root") {
      return childrenRPN.join(" ");
    } else {
      return [...childrenRPN, nodeString].join(" ");
    }
  } else {
    throw new Error("Invalid node type" + node.data);
  }
}

export function convertListToRPN(list: ListNode[]): string {
  let startState = true;
  let result = "";
  for (const node of list) {
    if (startState) {
      result += convertFeaturetoRPN(node.feature);
      if (node.negated) {
        result += " `NOT`";
      }
      startState = false;
    } else {
      result += " " + convertFeaturetoRPN(node.feature);
      if (node.negated) {
        result += " `NOT`";
      }
      result += " `AND`";
    }
  }
  return result;
}

export function convertTreeToDisplayString(phenotype: PhenotypeNode): string {
  const nodeString = convertNodeToRPN(phenotype);
  if (isFeature(phenotype.data) || isConstant(phenotype.data)) {
    return nodeString;
  } else if (isOperator(phenotype.data)) {
    const childrenDisplay = phenotype.children
      .map(convertTreeToDisplayString)
      .join(", ");
    if (phenotype.data.name === "Root") {
      return childrenDisplay;
    } else {
      return `${nodeString}(${childrenDisplay})`;
    }
  } else {
    throw new Error("Invalid node type" + phenotype.data);
  }
}

export function convertTreeToListOfCodes(node: PhenotypeNode): string[] {
  if (isFeature(node.data)) {
    return [node.data.code];
  } else if (isOperator(node.data)) {
    const childrenCodes = node.children.map(convertTreeToListOfCodes);
    return childrenCodes.flat();
  } else {
    return [];
  }
}
