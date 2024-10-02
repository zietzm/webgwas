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
    // Rheumatoid arthritis (M05) and Systemic lupus erythematosus (M32)
    ["M05", false, "M32", false],
    // Alzheimer's disease (G30) and Dementia (F03)
    ["G30", false, "F03", false],
    // Psoriasis (L40) and Psoriatic arthritis (M07)
    ["L40", false, "M07", false],
    // Type 1 Diabetes (E10) and Celiac disease (K90.0)
    ["E10", false, "K90.0", false],
    // Crohn's disease (K50) and Ulcerative colitis (K51)
    ["K50", false, "K51", false],
    // Osteoporosis (M81) and Hip fracture (S72)
    ["M81", false, "S72", false],
    // Parkinson's disease (G20) and Essential tremor (G25.0)
    ["G20", false, "G25.0", false],
    // Eczema (L30) and Asthma (J45)
    ["L30", false, "J45", false],
    // Multiple sclerosis (G35) and Optic neuritis (H46)
    ["G35", false, "H46", false],
    // Sickle cell disease (D57) and Thalassemia (D56)
    ["D57", false, "D56", false],
    // Breast cancer (C50) and Ovarian cancer (C56)
    ["C50", false, "C56", false],
    // Prostate cancer (C61) and Benign prostatic hyperplasia (N40)
    ["C61", false, "N40", false],
    // Colon cancer (C18) and Colon polyp (K63.5)
    ["C18", false, "K63.5", false],
    // Hemophilia (D66) and Coagulopathies (D68)
    ["D66", false, "D68", false],
    // Familial hypercholesterolemia (E78.0) and Coronary artery disease (I25)
    ["E78.0", false, "I25", false],
    // Schizophrenia (F20) and Bipolar disorder (F31)
    ["F20", false, "F31", false],
    // Glaucoma (H40) and Cataract (H25)
    ["H40", false, "H25", false],
    // Gout (M10) and Hyperuricemia (E79.0)
    ["M10", false, "E79.0", false],
    // Polycystic ovary syndrome (E28.2) and Infertility (N97)
    ["E28.2", false, "N97", false],
    // Macular degeneration (H35.3) and Retinal detachment (H33)
    ["H35.3", false, "H33", false],
    // Obstructive sleep apnea (G47.3) and Obesity (E66)
    ["G47.3", false, "E66", false],
    // Hypertension (I10) and Kidney disease (N18)
    ["I10", false, "N18", false],
    // Epilepsy (G40) and Migraine (G43)
    ["G40", false, "G43", false],
    // Autism spectrum disorder (F84.0) and Intellectual disabilities (F70)
    ["F84.0", false, "F70", false],
    // Amyotrophic lateral sclerosis (G12.2) and Frontotemporal dementia (G31.0)
    ["G12.2", false, "G31.0", false],
    // Endometriosis (N80) and Infertility (N97)
    ["N80", false, "N97", false],
    // Vitiligo (L80) and Autoimmune thyroiditis (E06.3)
    ["L80", false, "E06.3", false],
    // Hyperthyroidism (E05) and Atrial fibrillation (I48)
    ["E05", false, "I48", false],
    // Chronic kidney disease (N18) and Anemia (D50)
    ["N18", false, "D50", false],
    // Osteoarthritis (M15) and Obesity (E66)
    ["M15", false, "E66", false],
    // Cataract (H25) and Diabetes (E11)
    ["H25", false, "E11", false],
    // Barrett's esophagus (K22.7) and GERD (K21)
    ["K22.7", false, "K21", false],
    // Fibromyalgia (M79.7) and Chronic fatigue syndrome (G93.3)
    ["M79.7", false, "G93.3", false],
    // Deep vein thrombosis (I82) and Pulmonary embolism (I26)
    ["I82", false, "I26", false],
    // Chronic liver disease (K76) and Alcohol dependence (F10.2)
    ["K76", false, "F10.2", false],
    // Depression (F32) and Insomnia (G47.0)
    ["F32", false, "G47.0", false],
    // Anorexia nervosa (F50.0) and Bulimia nervosa (F50.2)
    ["F50.0", false, "F50.2", false],
    // ADHD (F90.0) and Learning disabilities (F81)
    ["F90.0", false, "F81", false],
    // Chronic pancreatitis (K86) and Diabetes mellitus (E11)
    ["K86", false, "E11", false],
    // Interstitial lung disease (J84) and Rheumatoid arthritis (M05)
    ["J84", false, "M05", false],
    // Myocardial infarction (I21) and Smoking (Z72.0)
    ["I21", false, "Z72.0", false],
    // Peripheral artery disease (I73.9) and Diabetes mellitus (E11)
    ["I73.9", false, "E11", false],
    // Chronic venous insufficiency (I87.2) and Varicose veins (I83)
    ["I87.2", false, "I83", false],
    // Sleep apnea (G47.3) and Hypertension (I10)
    ["G47.3", false, "I10", false],
    // Hyperparathyroidism (E21) and Osteoporosis (M81)
    ["E21", false, "M81", false],
    // Hepatitis B (B16) and Cirrhosis of liver (K74)
    ["B16", false, "K74", false],
    // Hepatitis C (B17.1) and Cirrhosis of liver (K74)
    ["B17.1", false, "K74", false],
    // Gallstones (K80) and Pancreatitis (K85)
    ["K80", false, "K85", false],
    // Bacterial meningitis (G00) and Sepsis (A41)
    ["G00", false, "A41", false],
    // HIV disease (B20) and Tuberculosis (A15)
    ["B20", false, "A15", false],
    // Leukemia (C91) and Anemia (D50)
    ["C91", false, "D50", false],
    // Cystic fibrosis (E84) and Pancreatic insufficiency (K86.8)
    ["E84", false, "K86.8", false],
    // Congenital heart disease (Q24) and Down syndrome (Q90)
    ["Q24", false, "Q90", false],
    // Phenylketonuria (E70.0) and Intellectual disabilities (F70)
    ["E70.0", false, "F70", false],
    // Scoliosis (M41) and Marfan syndrome (Q87.4)
    ["M41", false, "Q87.4", false],
    // Osteogenesis imperfecta (Q78.0) and Bone fractures (S82)
    ["Q78.0", false, "S82", false],
    // Hemochromatosis (E83.1) and Liver disease (K76)
    ["E83.1", false, "K76", false],
    // Obesity (E66) and Not diabetes mellitus (E11)
    ["E66", false, "E11", true],
    // Cholelithiasis (K80) and Obesity (E66)
    ["K80", false, "E66", false],
    // Iron-deficiency anemia (D50) and Chronic blood loss (D62)
    ["D50", false, "D62", false],
    // Psoriasis (L40) and Not psoriatic arthritis (M07)
    ["L40", false, "M07", true],
    // Systemic lupus erythematosus (M32) and Kidney involvement (N04)
    ["M32", false, "N04", false],
    // COPD (J44) and Not asthma (J45)
    ["J44", false, "J45", true],
    // Alcoholic liver disease (K70) and Pancreatitis (K85)
    ["K70", false, "K85", false],
    // Obesity (E66) and Not hypertension (I10)
    ["E66", false, "I10", true],
    // Vitamin D deficiency (E55.9) and Osteoporosis (M81)
    ["E55.9", false, "M81", false],
    // Ankylosing spondylitis (M45) and Uveitis (H20)
    ["M45", false, "H20", false],
    // Hypertrophic cardiomyopathy (I42.1) and Sudden cardiac death (I46.1)
    ["I42.1", false, "I46.1", false],
    // Cerebral palsy (G80) and Epilepsy (G40)
    ["G80", false, "G40", false],
    // Migraine (G43) and Depression (F32)
    ["G43", false, "F32", false],
    // Preeclampsia (O14) and Eclampsia (O15)
    ["O14", false, "O15", false],
    // Autism spectrum disorder (F84.0) and Epilepsy (G40)
    ["F84.0", false, "G40", false],
    // Diabetes mellitus (E11) and Peripheral neuropathy (G62.9)
    ["E11", false, "G62.9", false],
    // Cystic fibrosis (E84) and Male infertility (N46)
    ["E84", false, "N46", false],
    // Endometrial cancer (C54) and Obesity (E66)
    ["C54", false, "E66", false],
    // Hyperlipidemia (E78) and Not hypertension (I10)
    ["E78", false, "I10", true],
    // Inflammatory bowel disease (K50) and Primary sclerosing cholangitis (K83.0)
    ["K50", false, "K83.0", false],
    // Osteoarthritis (M15) and Not rheumatoid arthritis (M05)
    ["M15", false, "M05", true],
    // Hypertension (I10) and Not obesity (E66)
    ["I10", false, "E66", true],
    // Chronic kidney disease (N18) and Not diabetes mellitus (E11)
    ["N18", false, "E11", true],
    // Diabetes mellitus (E11) and Retinopathy (H35.0)
    ["E11", false, "H35.0", false],
    // Type 1 Diabetes (E10) and Autoimmune thyroiditis (E06.3)
    ["E10", false, "E06.3", false],
    // Hemophilia (D66) and Not von Willebrand disease (D68.0)
    ["D66", false, "D68.0", true],
    // Sickle cell disease (D57) and Leg ulcers (L97)
    ["D57", false, "L97", false],
    // Polycystic kidney disease (Q61) and Hypertension (I10)
    ["Q61", false, "I10", false],
    // Multiple sclerosis (G35) and Not neuromyelitis optica (G36.0)
    ["G35", false, "G36.0", true],
    // Osteoporosis (M81) and Vertebral fracture (S22)
    ["M81", false, "S22", false],
    // Hepatitis B (B16) and Not hepatitis C (B17.1)
    ["B16", false, "B17.1", true],
    // Rheumatoid arthritis (M05) and Not osteoarthritis (M15)
    ["M05", false, "M15", true],
    // Glaucoma (H40) and Family history of glaucoma (Z83.5)
    ["H40", false, "Z83.5", false],
    // Celiac disease (K90.0) and Dermatitis herpetiformis (L13.0)
    ["K90.0", false, "L13.0", false],
    // Huntington's disease (G10) and Depression (F32)
    ["G10", false, "F32", false],
    // Schizophrenia (F20) and Substance abuse (F19)
    ["F20", false, "F19", false],
    // Breast cancer (C50) and BRCA mutation (Z15.01)
    ["C50", false, "Z15.01", false],
    // Coronary artery disease (I25) and Not diabetes mellitus (E11)
    ["I25", false, "E11", true],
    // Congestive heart failure (I50) and Hypertension (I10)
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
      className="bg-blue-main hover:bg-blue-dark text-white text-l font-semibold py-1 px-4 rounded-lg flex items-center transition-colors"
    >
      <Shuffle className="mr-2" size={20} />
      I&apos;m Feeling Lucky
    </button>
  );
}
