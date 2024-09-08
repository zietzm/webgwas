import { PlusCircle, MinusCircle } from "lucide-react";
import { useState } from "react";
import { isFeature, isOperator, PhenotypeNode } from "../lib/types";

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
  onAdd,
  onRemove,
  editable,
}: {
  node: PhenotypeNode;
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
  } else {
    name = `Constant: ${node.data.value}`;
  }

  return (
    <div className="ml-4">
      <div className="flex items-center">
        <span className="mr-2">{name}</span>
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
            className="ml-2 p-1 rounded hover:bg-gray-200 active:bg-gray-300"
          >
            <MinusCircle size={20} />
          </button>
        )}
      </div>
      {node.children && node.children.length > 0 && (
        <div className="ml-4">
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
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
