import React, { useState } from "react";
import { Feature, Operator, PhenotypeNode } from "../lib/types";
import FuzzySelect from "./FuzzySelect";

export default function NodeSelector({
  features,
  operators,
  onSelect,
}: {
  features: Feature[];
  operators: Operator[];
  onSelect: (node: PhenotypeNode) => void;
}) {
  const [constantValue, setConstantValue] = useState("");
  const [constantError, setConstantError] = useState("");

  const handleConstantSubmit = () => {
    const num = parseFloat(constantValue);
    if (isNaN(num)) {
      setConstantError("Please enter a valid number");
    } else {
      onSelect({
        id: Date.now(),
        data: {
          value: num,
          type: "real",
        },
        children: [],
      });
      setConstantValue("");
      setConstantError("");
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <h3 className="text-lg font-semibold">Operators</h3>
      <div className="grid grid-cols-6 gap-2">
        {operators.map((op) => (
          <button
            key={op.id}
            onClick={() => {
              const phenotypeNode: PhenotypeNode = {
                id: Date.now(),
                data: op,
                children: [],
              };
              onSelect(phenotypeNode);
            }}
            className="bg-blue-light hover:bg-blue-mid text-blue-dark font-semibold py-2 px-4 rounded"
          >
            {op.name}
          </button>
        ))}
      </div>
      <h3 className="text-lg font-semibold">Constant</h3>
      <div className="flex flex-wrap gap-2 w-full">
        <input
          type="text"
          value={constantValue}
          onChange={(e) => setConstantValue(e.target.value)}
          className="p-2 border rounded"
          placeholder="Enter a number"
        />
        <button
          onClick={handleConstantSubmit}
          className="bg-green-main hover:bg-green-dark text-white font-semibold py-2 px-4 rounded w-20"
        >
          Add
        </button>
      </div>
      {constantError && <p className="text-red-500 text-sm">{constantError}</p>}
      <h3 className="text-lg font-semibold">Fields</h3>
      <FuzzySelect
        fuseThreshold={0.3} // Higher number is more lenient
        options={features}
        closeMenuOnSelect={false}
        onChange={(selectedOption) => {
          const node: PhenotypeNode = {
            id: Date.now(),
            data: selectedOption as Feature,
            children: [],
          };
          onSelect(node);
        }}
        value={null}
        getOptionLabel={(option) =>
          `${option!.name} [${option.code}] (N=${option.sample_size})`
        }
        getOptionValue={(option) => option.code}
        placeholder="Search for a field..."
      />
    </div>
  );
}
