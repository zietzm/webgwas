import React, { useState, useEffect } from "react";
import { Play, CheckCircle, Loader, XCircle, MinusCircle } from "lucide-react";
import Select from "react-select";
import { Cohort, Feature } from "../lib/types";
import {
  fetchCohorts,
  convertListToRPN,
  runGWAS,
  getResults,
  fetchFeatures,
} from "../lib/api";

const API_URL: string = process.env.NEXT_PUBLIC_API_URL || "";

export default function SimplePhenotypeBuilder() {
  const [cohorts, setCohorts] = useState<Cohort[]>([]);
  const [selectedCohort, setSelectedCohort] = useState<Cohort | null>(null);
  const [features, setFeatures] = useState<Feature[]>([]);
  const [phenotype, setPhenotype] = useState<Feature[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState("");
  const [jobStatus, setJobStatus] = useState<
    "submitting" | "queued" | "done" | "error" | null
  >(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

  // Fetch cohorts
  useEffect(() => {
    const doFetch = async () => {
      setIsLoading(true);
      try {
        const cohorts = await fetchCohorts(API_URL);
        setCohorts(cohorts);
      } catch (error) {
        let errorMessage = "Error fetching cohorts";
        if (error instanceof Error) {
          errorMessage = `Error fetching cohorts: ${error.message}`;
        }
        setError(errorMessage);
      }
      setIsLoading(false);
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
    };
    handleFetchFeatures();
  }, [selectedCohort]);

  async function handleRunGWAS() {
    if (!selectedCohort) {
      alert("Please select a cohort first.");
      return;
    }
    if (!phenotype.length) {
      alert("Please add at least one node to your phenotype.");
    }
    try {
      setJobStatus("submitting");
      const phenotypeDefinition = convertListToRPN(phenotype);
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

  return (
    <div className="bg-white shadow-md rounded-lg p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-4">Select Cohort</h2>
        <div className="flex flex-wrap gap-2">
          {cohorts.map((cohort) => (
            <button
              key={cohort.id}
              onClick={() => {
                setSelectedCohort(cohort);
                setPhenotype([]);
                setJobStatus(null);
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
      {phenotype.map((node, index) => (
        <div className="ml-4 flex-1 flex-row gap-2">
          {index > 0 && <b className="text-blue-600">AND </b>}
          {node.name} [{node.code}]
          <button
            onClick={() => {
              setPhenotype((prevPhenotype) => {
                const newPhenotype = [...prevPhenotype];
                newPhenotype.splice(index, 1);
                return newPhenotype;
              });
            }}
            className="ml-2 p-1 rounded hover:bg-gray-200 active:bg-gray-300"
          >
            <MinusCircle size={20} />
          </button>
        </div>
      ))}
      <br />
      {selectedCohort && jobStatus === null && (
        <Select
          options={features}
          closeMenuOnSelect={false}
          onChange={(selectedOption) => {
            setPhenotype([...phenotype, selectedOption as Feature]);
          }}
          value={null}
          getOptionLabel={(option) => `${option!.name} [${option!.code}]`}
          getOptionValue={(option) => `${option!.id}`}
          placeholder="Search for a field..."
          className="mb-2"
        />
      )}
      <div className="flex flex-wrap gap-4 mb-6">
        {phenotype.length > 0 && (
          <button
            onClick={() => {
              handleRunGWAS();
            }}
            className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors"
          >
            <Play className="mr-2" size={20} />
            Run GWAS
          </button>
        )}
        {jobStatus && (
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
        )}
        {jobStatus && jobStatus === "done" && (
          <div>
            <a href={downloadUrl!} download>
              <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors">
                <CheckCircle className="mr-2" size={20} />
                Download Results
              </button>
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
