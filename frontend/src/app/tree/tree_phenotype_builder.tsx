import React, { useState, useEffect, useCallback } from "react";
import {
  addChild,
  Cohort,
  Feature,
  operators,
  PhenotypeNode,
  popFromTree,
} from "@/app/lib/types";
import { convertTreeToDisplayString, fetchFeatures } from "@/app/lib/api";
import NodeSelector from "@/app/ui/NodeSelector";
import TreeNode from "@/app/tree/tree_node";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";

export default function TreePhenotypeBuilder({
  cohort,
  phenotype,
  setPhenotype,
  isMutable,
}: {
  cohort: Cohort;
  phenotype: PhenotypeNode;
  setPhenotype: React.Dispatch<React.SetStateAction<PhenotypeNode>>;
  isMutable: boolean;
}) {
  const [features, setFeatures] = useState<Feature[]>([]);
  const [selectedNode, setSelectedNode] = useState<PhenotypeNode | null>(null);
  const [showNodeSelector, setShowNodeSelector] = useState<boolean>(false);

  // Fetch features
  useEffect(() => {
    async function handleFetchFeatures() {
      if (cohort) {
        const cohortFeatures = await fetchFeatures(cohort);
        const binaryFeatures = cohortFeatures.filter((f) => f.type === "BOOL");
        setFeatures(binaryFeatures);
      }
    }
    handleFetchFeatures();
  }, [cohort]);

  const handleSelect = useCallback(
    (newNode: PhenotypeNode): void => {
      if (selectedNode === null) {
        return;
      }
      setPhenotype((prevPhenotype) => {
        const updatedPhenotype = JSON.parse(JSON.stringify(prevPhenotype));
        const result = addChild(updatedPhenotype, selectedNode.id, newNode);
        return result;
      });
      setSelectedNode(null);
      setShowNodeSelector(false);
    },
    [selectedNode, setPhenotype, setSelectedNode, setShowNodeSelector],
  );

  const handleRemove = useCallback(
    (idToPop: number): void => {
      setPhenotype((prevPhenotype) => {
        const newPhenotype = JSON.parse(JSON.stringify(prevPhenotype));
        popFromTree(newPhenotype, idToPop);
        return newPhenotype;
      });
    },
    [setPhenotype],
  );

  const onAdd = useCallback(
    (node: PhenotypeNode): void => {
      setSelectedNode(node);
      setShowNodeSelector(true);
    },
    [setSelectedNode, setShowNodeSelector],
  );

  const onRemove = useCallback(
    (id: number): void => {
      handleRemove(id);
    },
    [handleRemove],
  );

  return (
    <div className="flex flex-col gap-4">
      {phenotype && (
        <TreePhenotypeDisplay
          phenotype={phenotype}
          selectedNode={selectedNode}
          onAdd={onAdd}
          onRemove={onRemove}
          isMutable={isMutable}
        />
      )}
      {isMutable && showNodeSelector && (
        <Card className="text-inherit">
          <CardHeader>
            <h2>Insert a node beneath the selected node</h2>
          </CardHeader>
          <CardContent>
            <NodeSelector
              features={features}
              operators={operators.slice(1)}
              onSelect={handleSelect}
            />
          </CardContent>
          <CardFooter>
            <button
              onClick={() => {
                setShowNodeSelector(false);
                setSelectedNode(null);
              }}
              className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded w-full"
            >
              Close
            </button>
          </CardFooter>
        </Card>
      )}
      {!isMutable && (
        <>
          <h2>String representation of phenotype</h2>
          <div className="bg-secondary p-2 md:p-x-4 rounded-lg border">
            {convertTreeToDisplayString(phenotype)}
          </div>
        </>
      )}
    </div>
  );
}

function TreePhenotypeDisplay({
  phenotype,
  selectedNode,
  onAdd,
  onRemove,
  isMutable,
}: {
  phenotype: PhenotypeNode;
  selectedNode: PhenotypeNode | null;
  onAdd: (node: PhenotypeNode) => void;
  onRemove: (id: number) => void;
  isMutable: boolean;
}) {
  return (
    <div className="bg-secondary p-2 md:p-x-4 rounded-lg border">
      <div className="flex flex-col gap-0.5">
        <TreeNode
          node={phenotype}
          selectedNode={selectedNode}
          onAdd={onAdd}
          onRemove={onRemove}
          editable={isMutable}
        />
      </div>
    </div>
  );
}
