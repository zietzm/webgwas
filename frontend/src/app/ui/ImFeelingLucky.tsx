import { Shuffle } from "lucide-react";
import { Feature, ListNode } from "../lib/types";

function findFeatureByCode(features: Feature[], code: string): Feature | null {
  for (const feature of features) {
    if (feature.code === code) {
      return feature;
    }
  }
  return null;
}

export function ImFeelingLuckyList({
  features,
  setPhenotype,
}: {
  features: Feature[];
  setPhenotype: React.Dispatch<React.SetStateAction<ListNode[]>>;
}) {
  const hardcoded: (string | boolean)[][] = [
    ["I10", false, "E11", false],
    ["I25", false, "I21", false],
    ["E66", false, "E78", false],
    ["J45", false, "J30", false],
    ["F32", false, "F41", false],
    ["I63", false, "I67", true],
    ["J44", false, "J42", false],
    ["N39", false, "N30", false],
    ["E03", false, "E04", false],
    ["G43", false, "G44", false],
    ["K21", false, "K29", false],
    ["D50", false, "E61", false],
    ["M05", false, "M32", false],
    ["G30", false, "F03", false],
    ["L40", false, "M07", false],
    ["E10", false, "K90", false],
    ["K50", false, "K51", false],
    ["M81", false, "S72", false],
    ["G20", false, "G25", false],
    ["L30", false, "J45", false],
    ["G35", false, "H46", false],
    ["D57", false, "D56", false],
    ["C50", false, "C56", false],
    ["C61", false, "N40", false],
    ["C18", false, "K63", false],
    ["D66", false, "D68", false],
    ["E78", false, "I25", false],
    ["F20", false, "F31", false],
    ["H40", false, "H25", false],
    ["M10", false, "E79", false],
    ["E28", false, "N97", false],
    ["H35", false, "H33", false],
    ["G47", false, "E66", false],
    ["I10", false, "N18", false],
    ["G40", false, "G43", false],
    ["G12", false, "G31", false],
    ["N80", false, "N97", false],
    ["L80", false, "E06", false],
    ["E05", false, "I48", false],
    ["N18", false, "D50", false],
    ["M15", false, "E66", false],
    ["H25", false, "E11", false],
    ["K22", false, "K21", false],
    ["M79", false, "G93", false],
    ["I82", false, "I26", false],
    ["K76", false, "F10", false],
    ["F32", false, "G47", false],
    ["F50", false, "F50", false],
    ["K86", false, "E11", false],
    ["J84", false, "M05", false],
    ["I21", false, "Z72", false],
    ["I73", false, "E11", false],
    ["I87", false, "I83", false],
    ["G47", false, "I10", false],
    ["E21", false, "M81", false],
    ["B16", false, "K74", false],
    ["B17", false, "K74", false],
    ["K80", false, "K85", false],
    ["G00", false, "A41", false],
    ["C91", false, "D50", false],
    ["E84", false, "K86", false],
    ["M41", false, "Q87", false],
    ["Q78", false, "S82", false],
    ["E83", false, "K76", false],
    ["E66", false, "E11", true],
    ["K80", false, "E66", false],
    ["D50", false, "D62", false],
    ["L40", false, "M07", true],
    ["M32", false, "N04", false],
    ["J44", false, "J45", true],
    ["K70", false, "K85", false],
    ["E66", false, "I10", true],
    ["E55", false, "M81", false],
    ["M45", false, "H20", false],
    ["I42", false, "I46", false],
    ["G80", false, "G40", false],
    ["G43", false, "F32", false],
    ["F84", false, "G40", false],
    ["E11", false, "G62", false],
    ["E84", false, "N46", false],
    ["C54", false, "E66", false],
    ["E78", false, "I10", true],
    ["K50", false, "K83", false],
    ["M15", false, "M05", true],
    ["I10", false, "E66", true],
    ["N18", false, "E11", true],
    ["E11", false, "H35", false],
    ["E10", false, "E06", false],
    ["D66", false, "D68", true],
    ["D57", false, "L97", false],
    ["Q61", false, "I10", false],
    ["M81", false, "S22", false],
    ["B16", false, "B17", true],
    ["M05", false, "M15", true],
    ["H40", false, "Z83", false],
    ["K90", false, "L13", false],
    ["F20", false, "F19", false],
    ["I25", false, "E11", true],
    ["I50", false, "I10", false],
  ];
  const options: ListNode[][] = hardcoded
    .map((combination) => {
      const x = findFeatureByCode(features, combination[0] as string);
      const y = findFeatureByCode(features, combination[2] as string);
      if (x === null || y === null) {
        return null;
      }
      if (
        typeof combination[1] !== "boolean" ||
        typeof combination[3] !== "boolean"
      ) {
        throw new Error("Combination must be boolean");
      }
      const xFeature: ListNode = {
        feature: x,
        negated: combination[1],
      };
      const yFeature: ListNode = {
        feature: y,
        negated: combination[3],
      };
      return [xFeature, yFeature];
    })
    .filter((combination) => combination !== null);

  function getRandomSelection() {
    return options[Math.floor(Math.random() * options.length)];
  }

  function handleClick() {
    const randomSelection = getRandomSelection();
    setPhenotype(randomSelection);
  }

  return (
    <button
      onClick={handleClick}
      className="bg-primary hover:bg-blue-dark text-primary-foreground text-l font-semibold py-1 px-4 h-[38px] rounded-lg flex gap-2 items-center transition-colors text-nowrap"
    >
      <Shuffle size={20} />
      I&apos;m Feeling Lucky
    </button>
  );
}
