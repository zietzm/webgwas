import React, { useState, useEffect } from "react";

import { MinusCircle } from "lucide-react";

import { Cohort, Feature, ListNode } from "@/app/lib/types";
import { fetchFeatures } from "@/app/lib/api";
import { ImFeelingLuckyList } from "@/app/ui/ImFeelingLucky";
import FeatureSelect from "@/components/feature_select";

export default function SimplePhenotypeBuilder({
  cohort,
  phenotype,
  setPhenotype,
  isMutable,
}: {
  cohort: Cohort;
  phenotype: ListNode[];
  setPhenotype: React.Dispatch<React.SetStateAction<ListNode[]>>;
  isMutable: boolean;
}) {
  const [features, setFeatures] = useState<Feature[]>([]);

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

  return (
    <div className="flex flex-col gap-4">
      {phenotype && phenotype.length > 0 && (
        <PhenotypeBuilderDisplay
          phenotype={phenotype}
          setPhenotype={setPhenotype}
          isMutable={isMutable}
        />
      )}
      {isMutable && (
        <div className="flex flex-wrap md:flex-nowrap gap-4 justify-between">
          <div className="grow flex-1 min-w-[300px]">
            <FeatureSelect
              features={features}
              onChange={(selectedOption) => {
                if (selectedOption === null) {
                  return;
                }
                const node: ListNode = {
                  feature: selectedOption,
                  negated: false,
                };
                setPhenotype([...phenotype, node]);
              }}
            />
          </div>
          <ImFeelingLuckyList features={features} setPhenotype={setPhenotype} />
        </div>
      )}
    </div>
  );
}

function PhenotypeBuilderDisplay({
  phenotype,
  setPhenotype,
  isMutable,
}: {
  phenotype: ListNode[];
  setPhenotype: React.Dispatch<React.SetStateAction<ListNode[]>>;
  isMutable: boolean;
}) {
  function Negated({ index }: { index: number }) {
    const not = (
      <b
        className={`p-1 text-red-600 bg-red-100 rounded inline align-middle ${isMutable ? "hover:bg-red-200 active:bg-red-300" : ""}`}
      >
        NOT
      </b>
    );
    if (isMutable) {
      return (
        <button
          disabled={!isMutable}
          onClick={() => {
            setPhenotype((prevPhenotype) => {
              const newPhenotype = [...prevPhenotype];
              newPhenotype[index].negated = false;
              return newPhenotype;
            });
          }}
        >
          {not}
        </button>
      );
    } else {
      return <span>{not}</span>;
    }
  }

  function NegateButton({ index }: { index: number }) {
    if (isMutable) {
      return (
        <button
          onClick={() => {
            setPhenotype((prevPhenotype) => {
              const newPhenotype = [...prevPhenotype];
              newPhenotype[index].negated = true;
              return newPhenotype;
            });
          }}
        >
          <a className="p-1 text-red-600 bg-red-100 hover:bg-red-200 active:bg-red-300 rounded inline align-middle no-underline">
            Negate?
          </a>
        </button>
      );
    }
  }

  function RemoveButton({ index }: { index: number }) {
    if (isMutable) {
      return (
        <button
          onClick={() => {
            setPhenotype((prevPhenotype) => {
              const newPhenotype = [...prevPhenotype];
              newPhenotype.splice(index, 1);
              if (newPhenotype.length === 1 && newPhenotype[0].negated) {
                newPhenotype[0].negated = false;
              }
              return newPhenotype;
            });
          }}
          className="p-1 rounded hover:bg-gray-200 active:bg-gray-300 inline align-middle"
        >
          <MinusCircle size={20} />
        </button>
      );
    }
  }

  return (
    <div className="bg-secondary p-2 md:p-x-4 rounded-lg border">
      <div className="flex flex-col gap-0.5">
        {phenotype.map((node, index) => (
          <span
            key={index}
            className="space-x-1 md:space-x-2 align-middle leading-normal"
          >
            {index > 0 && (
              <b className="text-blue-dark inline align-middle">AND</b>
            )}
            {phenotype.length > 1 && node.negated && <Negated index={index} />}
            <span className="inline align-middle">
              {node.feature.name} [{node.feature.code}] (N=
              {node.feature.sample_size})
            </span>
            {phenotype.length > 1 && !node.negated && (
              <NegateButton index={index} />
            )}
            <RemoveButton index={index} />
          </span>
        ))}
      </div>
    </div>
  );
}
