import { PlusCircle, MinusCircle } from 'lucide-react';
import { useState } from 'react';
import { Feature, Operator, Constant, PhenotypeNode, isFeature, isOperator, isConstant } from '../lib/types';


const Tooltip = ({ children, text }: { children: React.ReactNode, text: string }) => {
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


export default function TreeNode ({ node, onAdd, onRemove }: { node: PhenotypeNode, onAdd: (node: PhenotypeNode) => void, onRemove: (node: PhenotypeNode) => void }) {
  const canAddChild = isOperator(node.data) && node.children.length < node.data.arity;
  const tooltipText = isOperator(node.data)
    ? `${node.data.name} can have ${node.data.arity} child${node.data.arity > 1 ? 'ren' : ''}`
    : `${node.data.type} node can't have children`;

  const nodeType = isFeature(node.data) ? 'Feature' : isOperator(node.data) ? 'Operator' : 'Constant';
  let nodeName: string;
  if (isFeature(node.data)) {
    nodeName = node.data.name;
  } else if (isOperator(node.data)) {
    nodeName = node.data.name;
  } else if (isConstant(node.data)) {
    nodeName = node.data.value.toString();
  } else {
    console.log(node.data);
    throw new Error('Invalid node type' + node.data.name);
  }

  return (
    <div className="ml-4">
      <div className="flex items-center">
        <span className="mr-2">{nodeName} ({nodeType})</span>
        <Tooltip text={tooltipText}>
          <button 
            onClick={() => canAddChild && onAdd(node)} 
            className={`p-1 rounded ${canAddChild 
              ? 'hover:bg-gray-200 active:bg-gray-300' 
              : 'opacity-50 cursor-not-allowed'}`}
            disabled={!canAddChild}
          >
            <PlusCircle size={20} />
          </button>
        </Tooltip>
        {node.id !== 0 && (
          <button 
            onClick={() => onRemove(node)} 
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
            />
          ))}
        </div>
      )}
    </div>
  );
};
