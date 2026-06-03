const samples = [
  // "melodic_orchestral_01.wav",
  "melodic_orchestral_02.wav",
  "melodic_orchestral_03.wav",
  // "melodic_orchestral_04.wav",
  "melodic_orchestral_05.wav",
  "melodic_orchestral_06.wav",
  "nonmelodic_01.wav",
  // "nonmelodic_02.wav",
  // "nonmelodic_03.wav",
  "nonmelodic_04.wav",
  "horror_01.wav",
  "horror_02.wav",
  "horror_03.wav",
  "east_asian_01.wav",
  "indian_01.wav",
  "indian_02.wav",
  "indian_03.wav",
  "upbeat_01.wav",
  "upbeat_02.wav",
  "upbeat_03.wav",
  "upbeat_04.wav"
];

const models = [
  "Sheetsage",
  "MT3",
  "YMT3+",
  "MR-MT3",
  "YPTF.MoE+Multi_noPS",
  "YPTF.MoE+Multi_PS",
  "YPTF+Multi_PS"
];

const table = document.getElementById("transcription-table");

const thead = document.createElement("thead");
const headerRow = document.createElement("tr");

const cornerTh = document.createElement("th");
cornerTh.textContent = "Model / Sample";
headerRow.appendChild(cornerTh);

samples.forEach(sample => {
  const th = document.createElement("th");
  th.textContent = sample;
  headerRow.appendChild(th);
});

thead.appendChild(headerRow);
table.appendChild(thead);

const tbody = document.createElement("tbody");

/* samples row with spectrogram players */
const samplesRow = document.createElement("tr");
const samplesHeader = document.createElement("th");
samplesHeader.textContent = "samples";
samplesHeader.classList.add("model-name");
samplesRow.appendChild(samplesHeader);

samples.forEach(() => {
  const td = document.createElement("td");

  const player = document.createElement("div");
  player.className = "sample-player";

  const waveformDiv = document.createElement("div");
  waveformDiv.className = "sample-waveform";

  const button = document.createElement("button");
  button.className = "play-toggle";
  button.textContent = "Play";

  player.appendChild(waveformDiv);
  player.appendChild(button);
  td.appendChild(player);
  samplesRow.appendChild(td);
});

tbody.appendChild(samplesRow);
//
// /* model rows (still show midi paths) */
// models.forEach(model => {
//   const row = document.createElement("tr");
//   const rowHeader = document.createElement("th");
//   rowHeader.textContent = model;
//   rowHeader.classList.add("model-name");
//   row.appendChild(rowHeader);
//
//   samples.forEach(sample => {
//     const td = document.createElement("td");
//     const midiName = sample.replace(/\.wav$/i, ".mid");
//     const path = `${model}/${midiName}`;
//
//     const pathWrapper = document.createElement("div");
//     pathWrapper.className = "path";
//
//     const link = document.createElement("a");
//     link.href = path;
//     link.textContent = path;
//     link.target = "_blank";
//     link.className = "path-main";
//
//     const badge = document.createElement("span");
//     badge.className = "badge";
//     badge.textContent = "midi";
//
//     pathWrapper.appendChild(link);
//     pathWrapper.appendChild(badge);
//     td.appendChild(pathWrapper);
//
//     row.appendChild(td);
//   });
//
//   tbody.appendChild(row);
// });
//
// table.appendChild(tbody);

/* model rows: html-midi-player + visualizer per MIDI file */
models.forEach((model, modelIndex) => {
  const row = document.createElement("tr");
  const rowHeader = document.createElement("th");
  rowHeader.textContent = model;
  rowHeader.classList.add("model-name");
  row.appendChild(rowHeader);

  samples.forEach((sample, sampleIndex) => {
    const td = document.createElement("td");
    const midiName = sample.replace(/\.wav$/i, ".mid");
    const path = `${model}/${midiName}`;

    const cell = document.createElement("div");
    cell.className = "midi-cell";

    const visEl = document.createElement("midi-visualizer");
    const visId = `vis-${modelIndex}-${sampleIndex}`;
    visEl.id = visId;
    visEl.setAttribute("type", "piano-roll");
    visEl.setAttribute("src", path);

    const playerEl = document.createElement("midi-player");
    playerEl.setAttribute("src", path);
    playerEl.setAttribute("sound-font", "");
    playerEl.setAttribute("visualizer", `#${visId}`);

    cell.appendChild(visEl);
    cell.appendChild(playerEl);
    td.appendChild(cell);
    row.appendChild(td);
  });

  tbody.appendChild(row);
});

table.appendChild(tbody);

/* WaveSurfer + Spectrogram instances */
const wavesurfers = [];

const sampleWaveforms = document.querySelectorAll(".sample-waveform");
const sampleButtons = document.querySelectorAll(".play-toggle");

sampleWaveforms.forEach((waveformEl, index) => {
  const sampleName = samples[index];
  const url = `samples/${sampleName}`;

  const ws = WaveSurfer.create({
    container: waveformEl,
    // visually hide the waveform, we only care about the spectrogram
    waveColor: "rgba(187,247,208,0.49)",
    progressColor: "#25fa70",
    cursorColor: "#bbf7d0",
    height: 50,
    url
  });

  const SpectrogramPlugin =
    (window.WaveSurfer && (WaveSurfer.Spectrogram || WaveSurfer.spectrogram)) || null;

  if (SpectrogramPlugin) {
    ws.registerPlugin(
      SpectrogramPlugin.create({
        container: waveformEl,
        labels: false,
        fftSamples: 2048
      })
    );
  } else {
    console.warn("Spectrogram plugin not found on WaveSurfer");
  }

  wavesurfers.push(ws);
});

sampleButtons.forEach((button, index) => {
  const ws = wavesurfers[index];

  button.addEventListener("click", () => {
    if (!ws) return;

    if (ws.isPlaying()) {
      ws.pause();
      button.textContent = "Play";
      return;
    }

    wavesurfers.forEach((other, i) => {
      if (other && other.isPlaying()) {
        other.pause();
        if (sampleButtons[i]) sampleButtons[i].textContent = "Play";
      }
    });

    ws.play();
    button.textContent = "Pause";
  });
});
document.querySelectorAll(".midi-cell midi-visualizer").forEach(vis => {
  vis.config = {
    noteHeight: 2,          // thinner notes
    pixelsPerTimeStep: 6,  // smaller = more compressed timeline
    // minPitch: 36,           // C2-ish
    // maxPitch: 96            // C7-ish
  };
});