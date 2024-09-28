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
    // Hypertension and type 2 diabetes
    ["I10", false, "E11", false],
    // Heart disease and heart attack
    ["I25", false, "I21", false],
    // Obesity (E66) and Hyperlipidemia (E78)
    ["E66", false, "E78", false],
    // Asthma (J45) and Allergic rhinitis (J30)
    ["J45", false, "J30", false],
    // Depression (F32) and Anxiety (F41)
    ["F32", false, "F41", false],
    // Stroke (I63) and Cerebrovascular disease (I67)
    ["I63", false, "I67", true],
    // Chronic obstructive pulmonary disease (COPD) (J44) and Chronic bronchitis (J42)
    ["J44", false, "J42", false],
    // Urinary tract infection (N39) and Cystitis (N30)
    ["N39", false, "N30", false], // bad
    // Hypothyroidism (E03) and Goiter (E04)
    ["E03", false, "E04", false],
    // Migraine (G43) and Tension-type headache (G44)
    ["G43", false, "G44", false], // bad
    // Gastroesophageal reflux disease (GERD) (K21) and Gastritis (K29)
    ["K21", false, "K29", false],
    // Anemia (D50) and Iron deficiency (E61)
    ["D50", false, "E61", false], // bad
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
      className="bg-blue-500 hover:bg-blue-700 text-white text-l font-semibold py-1 px-4 rounded-lg flex items-center transition-colors"
    >
      <Shuffle className="mr-2" size={20} />
      I'm Feeling Lucky
    </button>
  );
}
