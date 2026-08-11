# adada2026_con

This repository contains the stimulus materials used in the preliminary melody-evaluation experiment reported in the following paper:

> Ren Okumura, Kotaro Yoshizaki, Soma Arai, Hidefumi Ohmura, and Takuro Shibayama, “Extending Cognitive-Bias-Based Melody Generation to the Rhythm Domain: Perceptual Effects of Pitch and Rhythm Transformations,” ADADA 2026.

## Contents

- `musicxml/`: Original and transformed MusicXML scores.
- `wav/`: WAV stimuli rendered in MuseScore 4 using the Muse Sounds “Grand Piano” instrument.
- `assignments/`: CSV files specifying the WAV stimuli and presentation order assigned to each presentation set.
- `musicxml generation code/`: The Python source code used to generate the transformed MusicXML files
is provided in the `code/` directory. The generated files used in the
experiment are provided in `musicxml/`.
The labels P01–P30 are folder labels used to distinguish the 30 presentation sets. No participant responses, participant names, or other directly identifying information are included in this repository.

The assignment CSV files provide the information needed to identify the stimuli and their presentation order in each set. Stimulus filenames correspond to the original melody, pitch-transformation condition, rhythm-transformation condition, and generation seed.

## Version

Version 1.0 corresponds to the materials used in the experiment reported in the paper above.

## Citation

When using these materials, please cite the paper above.
