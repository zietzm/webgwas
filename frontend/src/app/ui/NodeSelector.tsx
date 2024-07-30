'use client'

import React, { useState } from 'react';
import { PlusCircle, MinusCircle, Play, CheckCircle, Loader, XCircle } from 'lucide-react';
import Select from 'react-select';

export default function NodeSelector ({ nodes, onSelect, onClose }) {
  const [showConstantInput, setShowConstantInput] = useState(false);
  const [constantValue, setConstantValue] = useState('');
  const [constantError, setConstantError] = useState('');

  const operators = nodes.filter(node => node.type === 'operator' && node.id !== 0);
  const fields = nodes.filter(node => node.type === 'field');

  const handleConstantSubmit = () => {
    const num = parseFloat(constantValue);
    if (isNaN(num)) {
      setConstantError('Please enter a valid number');
    } else {
      onSelect({ id: Date.now(), type: 'constant', name: num.toString() });
      setConstantValue('');
      setConstantError('');
      setShowConstantInput(false);
    }
  };

  const fieldOptions = fields.map(field => ({ value: field.id, label: field.name }));

  return (
    <div className="fixed right-0 top-0 h-full w-80 bg-white shadow-lg p-6 overflow-y-auto">
      <h2 className="text-2xl font-bold mb-6">Add Node</h2>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Operators</h3>
        <div className="grid grid-cols-2 gap-2">
          {operators.map((op) => (
            <button
              key={op.id}
              onClick={() => onSelect(op)}
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
            {constantError && <p className="text-red-500 text-sm mb-2">{constantError}</p>}
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
        <Select
          options={fieldOptions}
          onChange={(selectedOption) => onSelect(fields.find(f => f.id === selectedOption.value))}
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
  );
};
