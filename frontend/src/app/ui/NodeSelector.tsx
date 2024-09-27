import React, { useState, useEffect } from "react";
import Select from "react-select";
import { Feature, Operator, PhenotypeNode } from "../lib/types";
import FuzzySelect from "./FuzzySelect";

export default function NodeSelector({
  features,
  operators,
  onSelect,
  onClose,
}: {
  features: Feature[];
  operators: Operator[];
  onSelect: (node: PhenotypeNode) => void;
  onClose: () => void;
}) {
  const [showConstantInput, setShowConstantInput] = useState(false);
  const [constantValue, setConstantValue] = useState("");
  const [constantError, setConstantError] = useState("");
  const [headerHeight, setHeaderHeight] = useState(0);

  useEffect(() => {
    const header = document.querySelector("header");
    if (header) {
      setHeaderHeight(header.offsetHeight + 1);
    }
  }, []);

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
      setShowConstantInput(false);
    }
  };

  return (
    <div className="fixed right-0 top-0 w-full h-full bg-black bg-opacity-50 flex justify-end z-50">
      <div
        style={{
          top: `${headerHeight}px`,
          height: `calc(100% - ${headerHeight}px)`,
        }}
        className="w-80 bg-white shadow-lg p-6 overflow-y-auto absolute right-0"
      >
        <h2 className="text-2xl font-bold mb-6">Add Node</h2>
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Operators</h3>
          <div className="grid grid-cols-2 gap-2">
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
                className="bg-blue-100 hover:bg-blue-200 text-blue-800 font-semibold py-2 px-4 rounded"
              >
                {op.name}
              </button>
            ))}
          </div>
        </div>
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Constant</h3>
          {!showConstantInput ? (
            <button
              onClick={() => setShowConstantInput(true)}
              className="bg-green-100 hover:bg-green-200 text-green-800 font-semibold py-2 px-4 rounded w-full"
            >
              Add Constant
            </button>
          ) : (
            <div>
              <input
                type="text"
                value={constantValue}
                onChange={(e) => setConstantValue(e.target.value)}
                className="w-full p-2 border rounded mb-2"
                placeholder="Enter a number"
              />
              {constantError && (
                <p className="text-red-500 text-sm mb-2">{constantError}</p>
              )}
              <button
                onClick={handleConstantSubmit}
                className="bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded w-full"
              >
                Add
              </button>
            </div>
          )}
        </div>
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Fields</h3>
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
              `${option!.name} [${option!.code}] (N=${option!.sample_size})`
            }
            getOptionValue={(option) => `${option!.id}`}
            placeholder="Search for a field..."
            className="mb-2"
          />
        </div>
        <button
          className="mt-4 bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded w-full"
          onClick={onClose}
        >
          Close
        </button>
      </div>
    </div>
  );
}
