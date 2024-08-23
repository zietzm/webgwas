"use client";

import React, { useState, useEffect } from "react";
import { Play, CheckCircle, Loader, XCircle } from "lucide-react";
import NodeSelector from "./NodeSelector";
import TreeNode from "./TreeNode";
import { Node } from "../lib/Node";
import Image from "next/image";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

const rootNode: Node = {
  id: 0,
  type: "operator",
  name: "Root",
  minArity: 1,
  maxArity: 1,
  children: [],
};

interface ValidationResult {
  isValid: boolean;
  message: string;
}

export function phenotypeBuilderUsage() {
  return (
    <div className="p-5 border border-t-0 border-gray-200 dark:border-gray-700">
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        To get started, select a cohort from the buttons below. Then define the
        phenotype that interests you by clicking the "+" button and selecting
        the type of node you want to add.
      </p>
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        As an example, suppose we are interested hypertensive diabetes (diabetes
        and hypertension). To build this phenotype, we would select first the
        "AND" operator, then add both diabetes (E11) and hypertension (I10) as
        children.
      </p>
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        The resulting phenotype would look like this:
      </p>
      <Image
        unoptimized
        src={"/examplePT.png"}
        alt={"Example phenotype"}
        width={250}
        height={131}
      />
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        Once built, you can validate your phenotype and run the GWAS. Our server
        will then start the GWAS calculation and display the status of the job.
        Once complete, a download link to your results will appear.
      </p>
    </div>
  );
}

export default function PhenotypeBuilder() {
  const [cohorts, setCohorts] = useState([]);
  const [selectedCohort, setSelectedCohort] = useState(null as string | null);
  const [nodes, setNodes] = useState([]);
  const [tree, setTree] = useState(rootNode);
  const [showNodeSelector, setShowNodeSelector] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null as Node | null);
  const [validationResult, setValidationResult] = useState(
    null as ValidationResult | null,
  );
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobId, setJobId] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null as string | null);

  useEffect(() => {
    const fetchCohorts = async () => {
      try {
        const response = await fetch(`${API_URL}/api/cohorts`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });
        if (!response.ok) {
          throw new Error("Failed to fetch cohorts");
        }
        const data = await response.json();
        setCohorts(data);
        setIsLoading(false);
      } catch (error) {
        let errorMessage = "Failed to fetch cohorts";
        if (error instanceof Error) {
          errorMessage = `Failed to fetch cohorts: ${error.message}`;
        }
        setError(errorMessage);
        setIsLoading(false);
      }
    };

    fetchCohorts();
  }, []);

  useEffect(() => {
    if (selectedCohort) {
      fetchNodes(selectedCohort);
    }
  }, [selectedCohort]);

  const fetchNodes = async (selectedCohort: string) => {
    try {
      const searchParams = new URLSearchParams({ cohort_name: selectedCohort });
      const response = await fetch(
        `${API_URL}/api/nodes?${searchParams.toString()}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        },
      );
      if (!response.ok) {
        throw new Error("Failed to fetch nodes");
      }
      const data = await response.json();
      data.forEach((node: Node) => {
        if (node.minArity === null) node.minArity = 0;
        if (node.maxArity === null) node.maxArity = Infinity;
      });
      setNodes(data);
      setTree(rootNode);
    } catch (error) {
      let errorMessage = "Failed to fetch nodes";
      if (error instanceof Error) {
        errorMessage = `Failed to fetch nodes: ${error.message}`;
      }
      setError(errorMessage);
    }
  };

  const handleCohortSelect = (cohort: string) => {
    setSelectedCohort(cohort);
    setTree(rootNode);
    setValidationResult(null);
  };

  const handleAdd = (parentNode: Node) => {
    setSelectedNode(parentNode);
    setShowNodeSelector(true);
  };

  const handleRemove = (nodeToRemove: Node) => {
    const removeNodeRecursive = (currentNode: Node) => {
      if (currentNode.children) {
        currentNode.children = currentNode.children.filter(
          (child) => child.id !== nodeToRemove.id,
        );
        currentNode.children.forEach(removeNodeRecursive);
      }
    };

    setTree((prevTree) => {
      const newTree = { ...prevTree };
      removeNodeRecursive(newTree);
      return newTree;
    });
  };

  const handleSelect = (newNode: Node | null) => {
    if (!newNode) {
      return;
    }
    setTree((prevTree) => {
      function addNodeRecursive(currentNode: Node): Node {
        if (currentNode.id === selectedNode!.id) {
          return {
            ...currentNode,
            children: [
              ...(currentNode.children || []),
              { ...newNode!, id: Date.now(), children: [] },
            ],
          };
        }
        if (currentNode.children) {
          return {
            ...currentNode,
            children: currentNode.children.map(addNodeRecursive),
          };
        }
        return currentNode;
      }

      return addNodeRecursive(prevTree);
    });

    setShowNodeSelector(false);
  };

  const validateTree = (node: Node): ValidationResult => {
    if (node.type === "operator") {
      const childCount = node.children ? node.children.length : 0;
      if (childCount < node.minArity || childCount > node.maxArity) {
        return {
          isValid: false,
          message: `${node.name} (${node.type}) should have between ${node.minArity} and ${node.maxArity} children, but has ${childCount}.`,
        };
      }
    }

    if (node.children) {
      for (let child of node.children) {
        const childValidation = validateTree(child);
        if (!childValidation.isValid) {
          return childValidation;
        }
      }
    }

    return { isValid: true, message: "Phenotype definition is valid." };
  };

  const handleValidate = () => {
    const result = validateTree(tree);
    setValidationResult(result);
  };

  const convertToRPN = (node: Node): string => {
    switch (node.type) {
      case "constant":
        return node.name;
      case "operator":
        const childrenRPN = node.children.map(convertToRPN);
        if (node.name === "Root") {
          return childrenRPN.join(" ");
        } else {
          return [...childrenRPN, "`" + node.name + "`"].join(" ");
        }
      case "field":
        return '"' + node.name + '"';
      default:
        return node.name;
    }
  };

  const handleRunGWAS = async () => {
    if (!validationResult || !validationResult.isValid) {
      alert("Please validate the phenotype definition first.");
      return;
    }

    const rpn = convertToRPN(tree);
    try {
      setJobStatus("submitting");
      const response = await fetch(`${API_URL}/api/igwas`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          phenotype_definition: rpn,
          cohort_name: selectedCohort,
        }),
      });
      if (!response.ok) {
        throw new Error("Failed to run GWAS");
      }
      const result = await response.json();
      setJobId(result.request_id);
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
  };

  const pollJobStatus = async (requestId: string) => {
    try {
      const response = await fetch(`${API_URL}/api/igwas/status/${requestId}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (!response.ok) {
        throw new Error(
          `Failed to fetch job status response: ${response.status}`,
        );
      }
      const result = await response.json();
      if (result.status === "done") {
        setJobStatus("done");
        downloadResults(requestId);
      } else if (result.status === "error") {
        setJobStatus("error");
        console.error("GWAS job failed. Please try again.", result);
      } else {
        setTimeout(() => pollJobStatus(requestId), 5000); // Poll every 5 seconds
      }
    } catch (err) {
      console.error("Error polling job status:", err);
      setJobStatus("error");
    }
  };

  const downloadResults = async (requestId: string) => {
    try {
      const response = await fetch(`${API_URL}/api/igwas/results/${requestId}`);
      if (!response.ok) {
        throw new Error("Failed to fetch results");
      }
      const result = await response.json();
      setDownloadUrl(result.url);
    } catch (err) {
      console.error("Error downloading results:", err);
      alert("Failed to download results. Please try again.");
      setJobStatus("error");
    }
  };

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
              key={cohort}
              onClick={() => handleCohortSelect(cohort)}
              className={`py-2 px-4 rounded-full transition-colors ${
                selectedCohort === cohort
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-200 text-gray-800 hover:bg-gray-300"
              }`}
            >
              {cohort}
            </button>
          ))}
        </div>
      </div>
      {selectedCohort && (
        <>
          <div className="border border-gray-200 p-4 rounded-lg mb-6">
            {tree ? (
              <TreeNode node={tree} onAdd={handleAdd} onRemove={handleRemove} />
            ) : (
              <p className="text-gray-600">
                Select a cohort to start building your phenotype.
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-4 mb-6">
            <button
              onClick={handleValidate}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors"
              disabled={
                !tree || jobStatus === "running" || jobStatus === "queued"
              }
            >
              <CheckCircle className="mr-2" size={20} />
              Validate Phenotype
            </button>
            <button
              onClick={handleRunGWAS}
              className={`bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors ${
                !validationResult || !validationResult.isValid || jobStatus
                  ? "opacity-50 cursor-not-allowed"
                  : ""
              }`}
              disabled={
                !validationResult ||
                !validationResult.isValid ||
                jobStatus === "running" ||
                jobStatus === "queued"
              }
            >
              <Play className="mr-2" size={20} />
              Run GWAS
            </button>
            {jobStatus && (
              <div className="flex items-center bg-gray-100 p-2 rounded-lg">
                {(jobStatus === "submitting" ||
                  jobStatus === "queued" ||
                  jobStatus === "running") && (
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
                  {jobStatus === "running" && "GWAS job running..."}
                  {jobStatus === "done" && "GWAS job completed."}
                  {jobStatus === "error" &&
                    "GWAS job failed. Please try again."}
                </span>
              </div>
            )}
          </div>
          {validationResult && (
            <div
              className={`mb-6 p-4 rounded-lg ${validationResult.isValid ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}
            >
              {validationResult.message}
            </div>
          )}
          {jobStatus && jobStatus === "done" && (
            <div className="mb-6 p-4 rounded-lg bg-blue-100 text-blue-800">
              <p>
                Download results from{" "}
                <a
                  href={downloadUrl!}
                  download
                  className="underline hover:text-blue-600"
                >
                  {downloadUrl}
                </a>
              </p>
            </div>
          )}
          {showNodeSelector && (
            <NodeSelector
              nodes={nodes}
              onSelect={handleSelect}
              onClose={() => setShowNodeSelector(false)}
            />
          )}
        </>
      )}
    </div>
  );
}
