interface Node {
  id: number;
  type: string;
  name: string;
  minArity: number;
  maxArity: number;
  children: Node[];
}

export type { Node };
