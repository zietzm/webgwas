import React, { useState, useEffect } from "react";
import {
  Play,
  CheckCircle,
  Loader,
  XCircle,
  Download,
  BookOpenCheck,
} from "lucide-react";
import {
  addChild,
  Cohort,
  Feature,
  operators,
  PhenotypeNode,
  PhenotypeSummary,
  popFromTree,
} from "../lib/types";
import {
  fetchCohorts,
  runGWAS,
  getResults,
  fetchFeatures,
  getPhenotypeSummary,
  convertTreeToRPN,
  convertTreeToDisplayString,
  validatePhenotype,
  ValidationResponse,
} from "../lib/api";
import PhenotypeScatterPlots from "./PhenotypeSummary";
import TreeNode from "./TreeNode";
import NodeSelector from "./NodeSelector";

const API_URL: string = process.env.NEXT_PUBLIC_API_URL || "";

let startPhenotype: PhenotypeNode = {
  id: Date.now(),
  data: operators[0],
  children: [],
};

export default function ComplexPhenotypeBuilder() {
  const [cohorts, setCohorts] = useState<Cohort[]>([]);
  const [selectedCohort, setSelectedCohort] = useState<Cohort | null>(null);
  const [features, setFeatures] = useState<Feature[]>([]);
  const [phenotype, setPhenotype] = useState<PhenotypeNode>(startPhenotype);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState("");
  const [validationResult, setValidationResult] =
    useState<ValidationResponse | null>(null);
  const [summary, setSummary] = useState<PhenotypeSummary | null>(null);
  const [jobStatus, setJobStatus] = useState<
    "valid" | "submitting" | "queued" | "done" | "error" | null
  >(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [showNodeSelector, setShowNodeSelector] = useState<boolean>(false);
  const [selectedNode, setSelectedNode] = useState<PhenotypeNode | null>(null);

  // Fetch cohorts
  useEffect(() => {
    const doFetch = async () => {
      setIsLoading(true);
      try {
        const cohorts = await fetchCohorts(API_URL);
        setCohorts(cohorts);
        setSelectedCohort(cohorts[0]);
      } catch (error) {
        let errorMessage = "Error fetching cohorts";
        if (error instanceof Error) {
          errorMessage = `Error fetching cohorts: ${error.message}`;
        }
        setError(errorMessage);
      }
    };
    doFetch();
  }, []);

  // Fetch features
  useEffect(() => {
    const handleFetchFeatures = async () => {
      if (selectedCohort) {
        const cohortFeatures = await fetchFeatures(API_URL, selectedCohort);
        setFeatures(cohortFeatures);
      }
      setIsLoading(false);
    };
    handleFetchFeatures();
  }, [selectedCohort]);

  async function handleRunGWAS() {
    if (!selectedCohort) {
      alert("Please select a cohort first.");
      return;
    }
    if (phenotype === null) {
      alert("Please add at least one node to your phenotype.");
      return;
    }
    try {
      setJobStatus("submitting");
      const phenotypeDefinition = convertTreeToRPN(phenotype);
      const result = await runGWAS(
        API_URL,
        phenotypeDefinition,
        selectedCohort,
      );
      setJobStatus(result.status);
      pollJobStatus(result.request_id);
    } catch (err) {
      let errorMessage = "Error running GWAS";
      if (err instanceof Error) {
        errorMessage = `Error running GWAS: ${err.message}`;
      }
      alert(errorMessage);
      setJobStatus("error");
    }
  }

  async function handleValidatePhenotype() {
    if (!selectedCohort) {
      alert("Please select a cohort first.");
      return;
    }
    if (phenotype === null) {
      alert("Please add at least one node to your phenotype.");
      return;
    }
    try {
      const phenotypeDefinition = convertTreeToRPN(phenotype);
      const validationResult = await validatePhenotype(
        API_URL,
        phenotypeDefinition,
        selectedCohort,
      );
      const phenotypeSummary = await getPhenotypeSummary(
        API_URL,
        phenotypeDefinition,
        selectedCohort,
      );
      setSummary(phenotypeSummary);
      if (phenotypeSummary.rsquared === null) {
        setValidationResult({
          is_valid: false,
          message:
            "R-squared cannot be calculated for this phenotype (all values are the same). Please try a different definition.",
        });
        setJobStatus("error");
      } else {
        setValidationResult(validationResult);
        setJobStatus("valid");
      }
    } catch (err) {
      let errorMessage = "Error validating the phenotype";
      if (err instanceof Error) {
        errorMessage = `Error validating the phenotype: ${err.message}`;
      }
      alert(errorMessage);
      setJobStatus("error");
    }
  }

  async function pollJobStatus(requestId: string) {
    try {
      const result = await getResults(API_URL, requestId);
      if (result.status === "done") {
        setJobStatus("done");
        downloadResults(requestId);
      } else if (result.status === "error") {
        setJobStatus("error");
        console.error("GWAS job failed. Please try again.", result);
      } else {
        setTimeout(() => pollJobStatus(requestId), 1000); // Poll every second
      }
    } catch (err) {
      console.error("Error polling job status:", err);
      setJobStatus("error");
    }
  }

  async function downloadResults(requestId: string) {
    try {
      const result = await getResults(API_URL, requestId);
      setDownloadUrl(result.url);
    } catch (err) {
      console.error("Error downloading results:", err);
      alert("Failed to download results. Please try again.");
      setJobStatus("error");
    }
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader className="animate-spin" size={40} />
      </div>
    );
  }
  if (error) {
    return <div className="text-red-600 text-center">{error}</div>;
  }

  function CohortSelector() {
    return (
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-4">Select Cohort</h2>
        <div className="flex flex-wrap gap-2">
          {cohorts.map((cohort) => (
            <button
              key={cohort.id}
              onClick={() => {
                if (jobStatus === null || jobStatus === "done") {
                  setSelectedCohort(cohort);
                  setPhenotype(startPhenotype);
                  setJobStatus(null);
                  setSummary(null);
                } else {
                  alert("Please wait for the current job to finish.");
                }
              }}
              className={`py-2 px-4 rounded-full transition-colors ${
                selectedCohort === cohort
                  ? "bg-blue-600 text-white"
                  : "bg-gray-200 text-gray-800 hover:bg-gray-300"
              }`}
            >
              {cohort.name}
            </button>
          ))}
        </div>
      </div>
    );
  }

  function JobStatusBoxes() {
    return (
      <div className="flex items-center bg-gray-100 p-2 rounded-lg">
        {(jobStatus === "submitting" || jobStatus === "queued") && (
          <Loader className="animate-spin mr-2" size={20} />
        )}
        {jobStatus === "done" && (
          <CheckCircle className="mr-2 text-green-600" size={20} />
        )}
        {jobStatus === "error" && (
          <XCircle className="mr-2 text-red-600" size={20} />
        )}
        <span className="text-gray-700">
          {jobStatus === "queued" && "GWAS job queued..."}
          {jobStatus === "done" && "GWAS job completed."}
          {jobStatus === "error" && "GWAS job failed. Please try again."}
        </span>
      </div>
    );
  }

  function ValidationStatusBoxes() {
    if (validationResult === null) {
      return;
    }
    const isValid = validationResult.is_valid;
    const message = validationResult.message;
    return (
      <div className="flex items-center bg-gray-100 p-2 rounded-lg">
        {isValid && <CheckCircle className="mr-2 text-green-600" size={20} />}
        {!isValid && <XCircle className="mr-2 text-red-600" size={20} />}
        <span className="text-gray-700"> {message} </span>
      </div>
    );
  }

  function GWASButtons() {
    return (
      <div className="flex flex-wrap gap-4 mb-6">
        {jobStatus === null && (
          <button
            onClick={handleValidatePhenotype}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors"
          >
            <BookOpenCheck className="mr-2" size={20} />
            Validate Phenotype
          </button>
        )}
        {validationResult !== null && <ValidationStatusBoxes />}
        {jobStatus && jobStatus === "valid" && (
          <button
            onClick={handleRunGWAS}
            className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors"
          >
            <Play className="mr-2" size={20} />
            Run GWAS
          </button>
        )}
        {jobStatus && jobStatus !== "valid" && <JobStatusBoxes />}
        {jobStatus && jobStatus === "done" && (
          <div>
            <a href={downloadUrl!} download>
              <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors">
                <Download className="mr-2" size={20} />
                Download Results
              </button>
            </a>
          </div>
        )}
      </div>
    );
  }

  function handleSelect(newNode: PhenotypeNode): void {
    if (selectedNode === null || phenotype === null) {
      return;
    }
    setPhenotype(addChild(phenotype, selectedNode.id, newNode));
    setShowNodeSelector(false);
    setSelectedNode(null);
  }

  function handleRemove(idToPop: number): void {
    setPhenotype((previousPhenotype) => {
      const newPhenotype = { ...previousPhenotype };
      popFromTree(newPhenotype, idToPop);
      return newPhenotype;
    });
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-6 items-center">
      <CohortSelector />
      {selectedCohort && (
        <div className="mb-6">
          <div className="text-xl font-semibold mb-4">Phenotype tree</div>
          <TreeNode
            node={phenotype}
            onAdd={(node) => {
              setSelectedNode(node);
              setShowNodeSelector(true);
            }}
            onRemove={handleRemove}
            editable={jobStatus === null}
          />
        </div>
      )}
      {selectedCohort && jobStatus !== null && (
        <div className="mb-6">
          <div className="text-xl font-semibold mb-4">Parsed phenotype</div>
          {convertTreeToDisplayString(phenotype)}
        </div>
      )}
      {selectedCohort && phenotype.children.length > 0 && <GWASButtons />}
      {summary && (
        <div className="mb-6">
          <PhenotypeScatterPlots data={summary} />
        </div>
      )}
      {showNodeSelector && (
        <NodeSelector
          features={features}
          operators={operators.slice(1)}
          onSelect={handleSelect}
          onClose={() => setShowNodeSelector(false)}
        />
      )}
    </div>
  );
}
