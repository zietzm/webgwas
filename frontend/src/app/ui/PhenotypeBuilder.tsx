'use client'

import React, { useState, useEffect } from 'react';
import { Play, CheckCircle, Loader, XCircle } from 'lucide-react';
import NodeSelector from './NodeSelector';
import TreeNode from './TreeNode';
import { Node } from '../lib/Node';

const rootNode: Node = {id: 0, type: 'operator', name: 'Root', minArity: 1, maxArity: 1, children: []};

interface ValidationResult {
  isValid: boolean;
  message: string;
}

export default function PhenotypeBuilder() {
  const [cohorts, setCohorts] = useState([]);
  const [selectedCohort, setSelectedCohort] = useState(null as string | null);
  const [nodes, setNodes] = useState([]);
  const [tree, setTree] = useState(rootNode);
  const [showNodeSelector, setShowNodeSelector] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null as Node | null);
  const [validationResult, setValidationResult] = useState(null as ValidationResult | null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobId, setJobId] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null as string | null);


  useEffect(() => {
    const fetchCohorts = async () => {
      try {
        const response = await fetch(
            'http://localhost:8000/api/cohorts', {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Private-Network': 'true',
              },
            }
        );
        if (!response.ok) {
          throw new Error('Failed to fetch cohorts');
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
      const searchParams = new URLSearchParams({cohort_name: selectedCohort});
      const response = await fetch(
        `http://localhost:8000/api/nodes?${searchParams.toString()}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      if (!response.ok) {
        throw new Error('Failed to fetch nodes');
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
          (child) => child.id !== nodeToRemove.id
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
    if (node.type === 'operator') {
      const childCount = node.children ? node.children.length : 0;
      if (childCount < node.minArity || childCount > node.maxArity) {
        return {
          isValid: false,
          message: `${node.name} (${node.type}) should have between ${node.minArity} and ${node.maxArity} children, but has ${childCount}.`
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

    return { isValid: true, message: 'Phenotype definition is valid.' };
  };

  const handleValidate = () => {
    const result = validateTree(tree);
    setValidationResult(result);
  };

  const convertToRPN = (node: Node): string => {
    switch (node.type) {
      case 'constant':
        return node.name;
      case 'operator':
        const childrenRPN = node.children.map(convertToRPN);
        if (node.name === 'Root') {
          return childrenRPN.join(' ');
        } else {
          return [...childrenRPN, '`' + node.name + '`'].join(' ');
        }
      case 'field':
        return '"' + node.name + '"';
      default:
        return node.name;
    }
  };

  const handleRunGWAS = async () => {
    if (!validationResult || !validationResult.isValid) {
      alert('Please validate the phenotype definition first.');
      return;
    }

    const rpn = convertToRPN(tree);
    try {
      setJobStatus('submitting');
      const response = await fetch('http://localhost:8000/api/igwas', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ phenotype_definition: rpn, cohort_name: selectedCohort }),
      });
      if (!response.ok) {
        throw new Error('Failed to run GWAS');
      }
      const result = await response.json();
      setJobId(result.request_id);
      setJobStatus(result.status);
      console.log(result.status);
      pollJobStatus(result.request_id);
    } catch (err) {
      let errorMessage = "Error running GWAS";
      if (err instanceof Error) {
        errorMessage = `Error running GWAS: ${err.message}`;
      }
      alert(errorMessage);
      setJobStatus('error');
    }
  };

  const pollJobStatus = async (requestId: string) => {
    try {
      const response = await fetch(
          `http://localhost:8000/api/igwas/status/${requestId}`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          }
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch job status response: ${response.status}`);
      }
      const result = await response.json();
      if (result.status === 'done') {
        setJobStatus('done');
        downloadResults(requestId);
      } else if (result.status === 'error') {
        setJobStatus('error');
        alert('GWAS job failed. Please try again.');
      } else {
        console.log(result.status);
        setTimeout(() => pollJobStatus(requestId), 5000); // Poll every 5 seconds
      }
    } catch (err) {
      console.error('Error polling job status:', err);
      setJobStatus('error');
    }
  };

  const downloadResults = async (requestId: string) => {
    console.log(jobStatus);
    try {
      const response = await fetch(`http://localhost:8000/api/igwas/results/${requestId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch results');
      }
      const result = await response.json();
      setDownloadUrl(result.url);
    } catch (err) {
      console.error('Error downloading results:', err);
      alert('Failed to download results. Please try again.');
      setJobStatus('error');
    }
  };

  if (isLoading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Phenotype Builder</h1>
      <div className="mb-4">
        <h2 className="text-xl font-semibold mb-2">Select Cohort</h2>
        <div className="flex space-x-2">
          {cohorts.map((cohort) => (
            <button
              key={cohort}
              onClick={() => handleCohortSelect(cohort)}
              className={`py-2 px-4 rounded ${
                selectedCohort === cohort
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
            >
              {cohort}
            </button>
          ))}
        </div>
      </div>
      {selectedCohort && (
        <>
          <div className="border p-4 rounded">
            {tree ? (
              <TreeNode node={tree} onAdd={handleAdd} onRemove={handleRemove} />
            ) : (
              <p>Select a cohort to start building your phenotype.</p>
            )}
          </div>
          <div className="mt-4 mb-4 flex items-center">
            <button
              onClick={handleValidate}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded flex items-center mr-2"
              disabled={!tree || jobStatus === 'running' || jobStatus === 'queued'}
            >
              <CheckCircle className="mr-2" size={20} />
              Validate Phenotype
            </button>
            <button
              onClick={handleRunGWAS}
              className={`bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded flex items-center mr-2 ${
                (!validationResult || !validationResult.isValid || jobStatus) ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              disabled={!validationResult || !validationResult.isValid || jobStatus === 'running' || jobStatus === 'queued'}
            >
              <Play className="mr-2" size={20} />
              Run GWAS
            </button>
            {jobStatus && (
              <div className="flex items-center">
                {(jobStatus === 'submitting' || jobStatus === 'queued' || jobStatus === 'running') && (
                  <Loader className="animate-spin mr-2" size={20} />
                )}
                {jobStatus === 'done' && (
                  <CheckCircle className="mr-2" size={20} />
                )}
                {jobStatus === 'error' && (
                  <XCircle className="mr-2" size={20} />
                )}
                <span>
                  {jobStatus === 'queued' && 'GWAS job queued...'}
                  {jobStatus === 'running' && 'GWAS job running...'}
                  {jobStatus === 'done' && 'GWAS job completed. Downloading results...'}
                  {jobStatus === 'failed' && 'GWAS job failed. Please try again.'}
                </span>
              </div>
            )}
          </div>
          {validationResult && (
            <div className={`mb-4 p-4 rounded ${validationResult.isValid ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
              {validationResult.message}
            </div>
          )}
          {jobStatus && jobStatus === 'done' && (
            <div className="mb-4 p-4 rounded bg-blue-100 text-blue-700">
              <p>Download results from <a href={downloadUrl!} download className="underline">{downloadUrl}</a></p>
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
};
