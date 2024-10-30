import { convertListToRPN, convertTreeToRPN } from "@/app/lib/api";
import { getCodesTree, ListNode, PhenotypeNode } from "@/app/lib/types";

export type Phenotype = ListPhenotype | TreePhenotype;

export interface ListPhenotype {
  type: "list";
  value: ListNode[];
}

export interface TreePhenotype {
  type: "tree";
  value: PhenotypeNode;
}

export function isSingleField(phenotype: Phenotype): boolean {
  if (phenotype.type === "list") {
    return phenotype.value.length === 1;
  } else {
    return phenotype.value.children.length === 0;
  }
}

export function convertToRPN(phenotype: Phenotype): string {
  if (phenotype.type === "list") {
    return convertListToRPN(phenotype.value);
  } else {
    return convertTreeToRPN(phenotype.value);
  }
}

export function getCodes(phenotype: Phenotype): string[] {
  if (phenotype.type === "list") {
    return phenotype.value.map((p) => p.feature.code);
  } else {
    return getCodesTree(phenotype.value);
  }
}
