import React from "react";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

interface QualityGaugeProps {
  quality: number;
}

export default function QualityGauge({
  quality,
}: QualityGaugeProps): React.ReactElement {
  const percentage = quality * 100;

  const getColor = (value: number) => {
    if (value < 0.3) return "#ff4d4d";
    if (value < 0.7) return "#ffa500";
    return "#4caf50";
  };

  return (
    <div className="w-52">
      <CircularProgressbar
        value={percentage}
        text={`${percentage.toFixed(1)}%`}
        circleRatio={0.75}
        styles={buildStyles({
          rotation: 1 / 2 + 1 / 8,
          strokeLinecap: "butt",
          trailColor: "#eee",
          pathColor: getColor(quality),
          textColor: "#333",
          textSize: "16px",
        })}
      />
    </div>
  );
}
