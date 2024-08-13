import React, { useState, useEffect } from 'react';
import { Play, CheckCircle, Loader, XCircle } from 'lucide-react';
import NodeSelector from './NodeSelector';
import TreeNode from './TreeNode';
import { Cohort, Feature, Operator, PhenotypeNode, Constant, isFeature, isOperator, isConstant, operators } from '../lib/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL;

interface ValidationResult {
  isValid: boolean;
  message: string;
}

const rootOperator: Operator = {id: 0, name: 'Root', arity: 1, input_type: 'any', output_type: 'any'} as Operator;
const rootNode: PhenotypeNode = {id: 0, data: rootOperator, children: []} as PhenotypeNode;

export default function PhenotypeBuilder() {
  const [cohorts, setCohorts] = useState<Cohort[]>([]);
  const [selectedCohort, setSelectedCohort] = useState<Cohort | null>(null);
  const [features, setFeatures] = useState<Feature[]>([]);
  const [tree, setTree] = useState(rootNode);
  const [showNodeSelector, setShowNodeSelector] = useState(false);
  const [selectedNode, setSelectedNode] = useState<PhenotypeNode | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState("");
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobId, setJobId] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);


  useEffect(() => {
    const fetchCohorts = async () => {
      try {
        const response = await fetch(
            `${API_URL}/api/cohorts`, {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
              },
            }
        );
        if (!response.ok) {
          throw new Error('Failed to fetch cohorts');
        }
        const data = await response.json();
        setCohorts(data as Cohort[]);
        setIsLoading(false);
      } catch (error) {
        let errorMessage = "Error fetching cohorts";
        if (error instanceof Error) {
          errorMessage = `Error fetching cohorts: ${error.message}`;
        }
        setError(errorMessage);
        setIsLoading(false);
      }
    };

    fetchCohorts();
  }, []);

  useEffect(() => {
    if (selectedCohort) {
        fetchFeatures(selectedCohort);
    }
  }, [selectedCohort]);

  async function fetchFeatures(cohort: Cohort) {
    try {
      const searchParams = new URLSearchParams({cohort_id: cohort.id.toString()});
      const response = await fetch(
        `${API_URL}/api/features?${searchParams.toString()}`,
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
      setFeatures(data as Feature[]);
      setTree(rootNode);
    } catch (error) {
      let errorMessage = "Failed to fetch nodes";
      if (error instanceof Error) {
        errorMessage = `Failed to fetch nodes: ${error.message}`;
      }
      setError(errorMessage);
    }
  };

  const handleCohortSelect = (cohort: Cohort) => {
    setSelectedCohort(cohort);
    setTree(rootNode);
    setValidationResult(null);
  };

  const handleAdd = (parentNode: PhenotypeNode) => {
    setSelectedNode(parentNode);
    setShowNodeSelector(true);
  };

  const handleRemove = (nodeToRemove: PhenotypeNode) => {
    const removeNodeRecursive = (currentNode: PhenotypeNode) => {
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

  const handleSelect = (newNode: Feature | Operator | Constant | null) => {
    if (!newNode) {
      return;
    }
    setTree((prevTree) => {
      function addNodeRecursive(currentNode: PhenotypeNode): PhenotypeNode {
            if (currentNode.id === selectedNode!.id) {
                return {
                    ...currentNode,
                    children: [
                        ...(currentNode.children || []),
                        { id: Date.now(), children: [], data: newNode! },
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

  const validateTree = (node: PhenotypeNode): ValidationResult => {
    // TODO: Implement validation using the API
    return { isValid: true, message: 'Phenotype definition is valid.' };
  };

  const handleValidate = () => {
    const result = validateTree(tree);
    setValidationResult(result);
  };

  const convertToRPN = (node: PhenotypeNode): string => {
    if (isFeature(node.data)) {
      return '"' + node.data.name + '"';
    } else if (isOperator(node.data)) {
      const childrenRPN = node.children.map(convertToRPN);
      if (node.data.name === 'Root') {
        return childrenRPN.join(' ');
      } else {
        return [...childrenRPN, '`' + node.data.name + '`'].join(' ');
      }
    } else if (isConstant(node.data)) {
      return node.data.value.toString();
    } else {
      throw new Error('Invalid node type' + node.data);
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
      const response = await fetch(`${API_URL}/api/igwas`, {
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
          `${API_URL}/api/igwas/status/${requestId}`, {
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
        console.error('GWAS job failed. Please try again.', result);
      } else {
        setTimeout(() => pollJobStatus(requestId), 5000); // Poll every 5 seconds
      }
    } catch (err) {
      console.error('Error polling job status:', err);
      setJobStatus('error');
    }
  };

  const downloadResults = async (requestId: string) => {
    try {
      const response = await fetch(`${API_URL}/api/igwas/results/${requestId}`);
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
    return <div className="flex justify-center items-center h-64"><Loader className="animate-spin" size={40} /></div>;
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
              onClick={() => handleCohortSelect(cohort)}
              className={`py-2 px-4 rounded-full transition-colors ${
                selectedCohort === cohort
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
              }`}
            >
              {cohort.name}
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
              <p className="text-gray-600">Select a cohort to start building your phenotype.</p>
            )}
          </div>
          <div className="flex flex-wrap gap-4 mb-6">
            <button
              onClick={handleValidate}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors"
              disabled={!tree || jobStatus === 'running' || jobStatus === 'queued'}
            >
              <CheckCircle className="mr-2" size={20} />
              Validate Phenotype
            </button>
            <button
              onClick={handleRunGWAS}
              className={`bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors ${
                (!validationResult || !validationResult.isValid || jobStatus) ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              disabled={!validationResult || !validationResult.isValid || jobStatus === 'running' || jobStatus === 'queued'}
            >
              <Play className="mr-2" size={20} />
              Run GWAS
            </button>
            {jobStatus && (
              <div className="flex items-center bg-gray-100 p-2 rounded-lg">
                {(jobStatus === 'submitting' || jobStatus === 'queued' || jobStatus === 'running') && (
                  <Loader className="animate-spin mr-2" size={20} />
                )}
                {jobStatus === 'done' && (
                  <CheckCircle className="mr-2 text-green-600" size={20} />
                )}
                {jobStatus === 'error' && (
                  <XCircle className="mr-2 text-red-600" size={20} />
                )}
                <span className="text-gray-700">
                  {jobStatus === 'queued' && 'GWAS job queued...'}
                  {jobStatus === 'running' && 'GWAS job running...'}
                  {jobStatus === 'done' && 'GWAS job completed.'}
                  {jobStatus === 'error' && 'GWAS job failed. Please try again.'}
                </span>
              </div>
            )}
          </div>
          {validationResult && (
            <div className={`mb-6 p-4 rounded-lg ${validationResult.isValid ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
              {validationResult.message}
            </div>
          )}
          {jobStatus && jobStatus === 'done' && (
            <div className="mb-6 p-4 rounded-lg bg-blue-100 text-blue-800">
              <p>Download results from <a href={downloadUrl!} download className="underline hover:text-blue-600">{downloadUrl}</a></p>
            </div>
          )}
          {showNodeSelector && (
            <NodeSelector
              features={features}
              operators={operators.filter(op => op.id !== 0)}
              onSelect={handleSelect}
              onClose={() => setShowNodeSelector(false)}
            />
          )}
        </>
      )}
    </div>
  );
}
