import { PlusCircle, MinusCircle } from "lucide-react";
import { useState } from "react";
import { isConstant, isFeature, isOperator, PhenotypeNode } from "../lib/types";

const Tooltip = ({
  children,
  text,
}: {
  children: React.ReactNode;
  text: string;
}) => {
  const [isVisible, setIsVisible] = useState(false);
  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
      >
        {children}
      </div>
      {isVisible && (
        <div className="absolute z-10 px-3 py-2 text-sm font-medium text-white bg-gray-900 rounded-lg shadow-sm tooltip dark:bg-gray-700 left-full ml-2 top-1/2 transform -translate-y-1/2 whitespace-nowrap">
          {text}
        </div>
      )}
    </div>
  );
};

export default function TreeNode({
  node,
  selectedNode,
  onAdd,
  onRemove,
  editable,
}: {
  node: PhenotypeNode;
  selectedNode: PhenotypeNode | null;
  onAdd: (node: PhenotypeNode) => void;
  onRemove: (id: number) => void;
  editable: boolean;
}) {
  const canAddChild =
    isOperator(node.data) && node.children.length < node.data.arity;
  let tooltipText: string;
  if (isOperator(node.data)) {
    tooltipText = `${node.data.name} can have ${node.data.arity} child${node.data.arity > 1 ? "ren" : ""}`;
  } else if (isFeature(node.data)) {
    tooltipText = "Feature cannot have children";
  } else {
    tooltipText = "Constant cannot have children";
  }
  let name = null;
  if (isOperator(node.data)) {
    name = node.data.name;
  } else if (isFeature(node.data)) {
    name = `${node.data.name} [${node.data.code}]`;
  } else if (isConstant(node.data)) {
    name = `Constant: ${node.data.value}`;
  } else {
    console.error(node.data);
    throw new Error("Invalid node type");
  }

  return (
    <div className="flex flex-col">
      <span
        className={`flex flex-wrap gap-2 items-center ${selectedNode && selectedNode.id === node.id ? "bg-gray-200 rounded-lg font-bold" : ""}`}
      >
        <div className="flex flex-row">
          {node.id !== 0 && (
            <div className="w-6 relative">
              <svg
                className="absolute top-1/2 -translate-y-1/2"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinejoin="round"
                strokeLinecap="round"
              >
                <path d="M 24 12 H 14 C 13 12 12 11 12 10 V 0" />
              </svg>
            </div>
          )}
          <span className="px-1">{name}</span>
        </div>
        {editable && (
          <Tooltip text={tooltipText}>
            <button
              onClick={() => canAddChild && onAdd(node)}
              className={`p-1 rounded ${
                canAddChild
                  ? "hover:bg-gray-200 active:bg-gray-300"
                  : "opacity-50 cursor-not-allowed"
              }`}
              disabled={!canAddChild}
            >
              <PlusCircle size={20} />
            </button>
          </Tooltip>
        )}
        {editable && !(isOperator(node.data) && node.data.name == "Root") && (
          <button
            onClick={() => onRemove(node.id)}
            className="p-1 rounded hover:bg-gray-200 active:bg-gray-300"
          >
            <MinusCircle size={20} />
          </button>
        )}
      </span>
      {node.children && node.children.length > 0 && (
        <div className={node.id === 0 ? "" : "ml-4"}>
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              selectedNode={selectedNode}
              onAdd={onAdd}
              onRemove={onRemove}
              editable={editable}
            />
          ))}
        </div>
      )}
    </div>
  );
}
