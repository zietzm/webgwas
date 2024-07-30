import { PlusCircle, MinusCircle } from 'lucide-react';
import { useState } from 'react';
import { Node } from '../lib/Node';


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


export default function TreeNode ({ node, onAdd, onRemove }: { node: Node, onAdd: (node: Node) => void, onRemove: (node: Node) => void }) {
  const canAddChild = node.type === 'operator' && node.children.length < node.maxArity;
  const tooltipText = node.type === 'operator' 
    ? `${node.name} can have ${node.maxArity} child${node.maxArity > 1 ? 'ren' : ''}`
    : `${node.type} node can't have children`;

  return (
    <div className="ml-4">
      <div className="flex items-center">
        <span className="mr-2">{node.name} ({node.type})</span>
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
