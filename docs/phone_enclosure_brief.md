# Design brief: phone camera enclosure and mount for football match capture

You are designing a printable enclosure and mount for a smartphone. The phone films a whole football (soccer) match from the sideline. The footage feeds a computer-vision pipeline that turns broadcast-style video into top-down player positions. The phone is a stand-in test camera: a bespoke camera assembly will replace it later. So the mounting interface you design now must carry over to that future camera.

This brief covers what the footage must look like, the range of phones to fit, heat and cold, stability, weather, power, manufacturing, what to hand back, and how the design will be tested. Where a number is given, treat it as a requirement unless it is marked as a target or a suggestion.

---

## 1. How the footage is used, and why it constrains the design

The pipeline uses the video to work out where the camera is looking:
1. It finds the white pitch lines.
2. It fits a model of the pitch to them.
3. It follows small camera movements from frame to frame using the texture of the grass.

From that it converts every player's position in the image to metres on the pitch. Field testing so far has shown four things that matter for the enclosure:

- **Wobble becomes position error.** The camera tracker reads any movement of the phone as the pitch moving. Slow sway or vibration adds noise to every player's position. A rigid mount matters more than anything else on this list.
- **Glare and flare break line detection.** Sun shining into the lens, or reflections off a window, wash out the pitch lines that calibration depends on.
- **The view must be elevated and see pitch lines.** Calibration is most accurate when the camera looks down on the pitch from height and can see several lines (touchlines, halfway line, penalty boxes).
- **Repeatable placement saves work.** If the phone goes back in exactly the same pose at the same venue, the calibration from last time can be reused.

The pipeline already corrects ordinary barrel distortion from ultra-wide lenses, so the ultra-wide lens is allowed and expected. Whatever is in front of the lens must not add distortion, haze or reflections.

---

## 2. Use case and environment

| Item | Requirement |
|---|---|
| Subject | A full-size pitch, 105 × 68 m (smaller youth pitches also occur) |
| Position | Elevated, on the sideline near the halfway line, looking across the pitch |
| Mounting | Light-stand or tripod mast (about 2–6 m), clamp on a stand railing or fence, or a bracket on a gantry or scaffold |
| Height above pitch | 4–15 m (higher is better; aim for at least 6 m) |
| Downward tilt | Adjustable about 10–35° below horizontal |
| Session length | Unattended for at least 2.5 h (90 min match, half-time, warm-up, margin) |
| Ambient | −5 to 40 °C air temperature, full sun, rain, wind to 15 m/s gusts |
| Venues | Grassroots (no power, public access) through stadium (crowd vibration on stands) |
| Operator | One person, setting up in under 5 minutes, possibly in gloves or rain |

### Coverage geometry (for sizing the field of view and the hood)
- A phone ultra-wide lens is about 13 mm equivalent focal length. That gives roughly **108° horizontal × 75° vertical in 16:9 video, about 120° diagonal**.
- From the halfway line, seeing both near corners needs the half-angle `atan(52.5 / d)` to be under 54°, where `d` is the horizontal distance back from the near touchline. That needs **d ≥ about 40 m**.
- Closer than that, the corners fall out of frame. This is acceptable: the pipeline handles a partial view. But it is why the **full ultra-wide field of view must never be clipped** by the enclosure.

---

## 3. Optical requirements (the most important section)

1. **No vignetting at any lens.**
   - The window opening, the hood and every part of the enclosure must stay outside a cone of at least **115° horizontal × 85° vertical**: the ultra-wide field of view plus 5° margin per side.
   - Measure the cone from the lens entrance, the front glass of the camera module.
   - Size the window aperture for the widest lens of the largest supported phone.
2. **Window.**
   - Flat, optical-grade clear acrylic (PMMA) or polycarbonate, 2–3 mm thick, ideally anti-reflective and anti-fog coated.
   - Mounted **parallel** to the phone's back glass (no tilt) and **as close to the lens as possible**, at most 2 mm air gap. This keeps the aperture small and avoids ghost reflections.
   - Replaceable: it will scratch.
   - State the cut dimensions so it can be laser-cut.
3. **Light seal around the lens.**
   - A matte-black foam or TPU gasket must ring the camera module and press against the phone back. This stops light from the screen, the flash LED, the status LEDs and the inside of the enclosure from reflecting into the lens off the window.
   - Cover or block the **flash** completely.
4. **Sun hood / flag.**
   - A hood above the window, extending forward, shades the lens from a high sun.
   - Its edge must stay outside the field-of-view cone in item 1.
   - Matte black inside.
   - Optional: an adjustable side flag for low sun.
5. **No fence mesh in view.**
   - When clamped behind a fence or railing, the lens must sit in front of the mesh, or close enough to it that the mesh is out of focus and outside the field of view.
   - Provide stand-off options of 0–80 mm.
6. **Know where the lens is.**
   - Document the position of the lens (the camera module centre) relative to the mount datum (the tripod screw axis and the plate face) for each supported phone, in a table.
   - The future bespoke camera will be placed on the same datum.

---

## 4. Phones to fit

Design to an **envelope**, not one phone. Phones go in **without** their own cases.

| Class (examples) | L × W × T (mm), approx. | Camera module |
|---|---|---|
| Small: iPhone SE / 13 mini class | 131–139 × 64–68 × 7.3–7.7 | top-left square cluster |
| Standard: iPhone 15/16/17, Galaxy S2x, Pixel 9/10 | 146–153 × 70–73 × 7.8–8.5 | iPhone: top-left cluster; Samsung: vertical column at top-left; Pixel: full-width horizontal bar |
| Large: iPhone Pro Max, Galaxy S2x Ultra, Pixel Pro XL | 160–165 × 76–79 × 8.2–8.9 | as above |

- **Camera bump:** adds 1.5–5 mm on top of T and sits off-centre. The cradle must not rock on the bump; support the phone on its flat back.
- **Envelope to cover:** L 130–166 mm, W 64–80 mm, T 7–9.5 mm plus bump up to 5 mm.
- **Out of scope:** foldables, tablets, and phones in thick cases.
- **Orientation:** landscape, either way up (lens on the left or the right). The design must allow both, and record which one is in use.

### Recommended fit approach (you may propose better)
- A parametric cradle that clamps the phone on its long edges, with a screw or spring jaw on one side. Cover a range of sizes with printed **shims or inserts** rather than one fixed pocket.
- Clamp on the frame edges, away from the side buttons, with thin TPU pads for grip.
- A lens window that covers every camera position in the envelope. Either use one large window with a movable light-seal gasket, or interchangeable window plates per phone family.
- Expose these parameters at the top of the model:
  - `phone_length`, `phone_width`, `phone_thickness`;
  - `bump_depth`;
  - `lens_center_x`, `lens_center_y` (measured from the phone's top-left corner, screen facing away);
  - `lens_module_diameter`;
  - `orientation`.

---

## 5. Heat and cold

Phones are rated for about **0–35 °C ambient** while operating. Recording video, especially 4K, while charging, in direct sun, inside a box, is the worst case. The phone will throttle, dim the screen, or **stop recording**. Thermal design is a primary requirement, not a nice-to-have.

**Requirements:**
- **Reflect and shade.**
  - Light-coloured exterior (white or light grey).
  - A separate **sun roof with an air gap** of at least 10 mm above the enclosure body, so the sun heats the roof, not the box.
- **Ventilate by convection.**
  - Inlet vents low, outlet vents high, so air flows like a chimney past the phone's back.
  - Vents are louvred or baffled so rain can't enter directly. **No sealed box.**
- **Pull heat from the phone back.**
  - Either leave the back open to the airflow (preferred), or press it against an aluminium spreader plate that is exposed to outside air, with a thermal pad.
  - Don't trap the phone between plastic walls.
- **Optional active cooling.**
  - A mount for a 30–40 mm 5 V USB fan blowing across the phone back, powered from the power bank.
  - It must not vibrate the phone. Soft-mount the fan, or mount it on the roof rather than the cradle.
- **Keep the power bank out of the phone's airflow.** It also gets warm while charging.
- **Cold:**
  - Batteries lose capacity below about 0 °C. Keep the power bank inside the enclosure.
  - Allow optional clip-in foam insulation panels for winter, with the vents closable.
- **Condensation:**
  - Anti-fog window (coating or film).
  - A pocket for a small desiccant sachet.
  - Drain holes at the lowest points in both landscape orientations.

**Materials:**
- **ASA preferred:** UV-stable, softens around 95–100 °C.
- **PETG acceptable:** less UV-stable.
- **Not PLA:** it softens around 55–60 °C, which a sunny car or sun-soaked enclosure can reach.
- **TPU** for gaskets, grip pads and jaw liners.

---

## 6. Stability and rigidity

- **Target:** under about 0.05° of angular shake in 10 m/s wind, and no measurable slow drift over 2.5 h.
  - For scale: 0.05° is about 1 pixel at 1080p with the ultra-wide lens.
  - Drift typically comes from creep, clamps loosening, or heat softening the plastic.
- **Stiff load path.** Phone → cradle → body → mount should be one short, deep, ribbed structure, with no long thin arms or cantilevers.
- **Centre of mass** (phone + power bank + enclosure) directly above the tripod screw axis.
- **Low wind profile.** Keep the frontal area small, round the edges, and don't build a big flat sail. The sun hood is the main sail risk: make it short, stiff and vented.
- **No soft mounts in the load path.** Rubber isolation adds sway. TPU pads are for grip only, at 1 mm or less compressed. Ground-borne vibration (crowd bouncing on a stand) is handled by mounting location and operator guidance, not by springing the phone.
- **Anti-rotation.**
  - Every screw interface needs a second locating feature (pin, key or dovetail stop), so the enclosure can't twist on a single 1/4" screw.
  - Clamp jaws need a positive lock.
- **Repeatable remounting.**
  - Keyed mating features and detents for tilt.
  - Engraved angle scale every 5°.
  - A small bubble level (or a pocket for a 10–15 mm vial) on top for roll.
  - Goal: unmount and remount at the same venue with the view moving no more than a few pixels.

---

## 7. Mounting interfaces

Provide on the enclosure base:
- **1/4"-20 UNC** threaded insert (brass heat-set) on the centre-of-mass axis, plus a **3/8"-16** insert.
- An **Arca-Swiss compatible dovetail** (38 mm width standard), with a safety stop at each end.
- An **anti-twist pin hole** (Arca and tripod plates often have a pin) 1/4" from the screw axis, or a recess for a common anti-twist plate.

Separate printed adapters that bolt to that base:
- A **round-tube clamp** for 25–50 mm poles and railings: two-part, M5 bolts, TPU liner, anti-rotation.
- A **flat-rail clamp** for rectangular rails and fence posts up to 60 mm.
- A **tilt head** for when no tripod head is used: 0–40° down tilt, with detents every 5° and a single locking knob that tightens by hand.

**Future-proofing:** the base interface (screw axis, dovetail, pin) is the contract. The bespoke camera will reuse it, so put the base's key dimensions and its datum in the documentation.

---

## 8. Power, cables and access

- **Power bank compartment.**
  - Fits a 20,000 mAh bank, about 150 × 70 × 25 mm (make it parametric).
  - Located low, near the mount axis, for stability.
  - Strapped or clamped so it can't rattle.
- **USB-C to the phone.**
  - Routing from the bank to the phone's port in **both** landscape orientations.
  - A strain-relief clamp so a tugged cable can't move the phone.
  - Space for a right-angle USB-C plug, about 25 × 15 mm.
- **External cable port** (for mains or an external bank), sealed with a grommet.
- **Operator access:**
  - Starting and stopping recording, and checking the screen, must be possible without removing the phone.
  - Provide a hinged or latched lid, or a viewing window with a sliding shutter, so the screen can be covered or turned off during recording.
  - Side buttons must not be pressed by the clamp. Either leave them accessible or keep them clear.
- **Lock and tether.**
  - A cable-lock slot (to deter theft at public pitches).
  - A tether loop on the body, so a failed clamp doesn't drop the phone.

---

## 9. Weather and durability

- **Rain:**
  - Splash and shower resistant, not submersible.
  - The roof overhangs the window and the vents.
  - Drip edges.
  - Gasketed lid.
  - Drain paths in both orientations.
- **Sun:** UV-stable material (ASA), light colour.
- **Impact:** survives a fall from 1.5 m onto grass and a knock against a railing without the phone coming out.
- **Hardware:** stainless M3, M4 and M5 screws with heat-set brass inserts, so threads survive repeated assembly.

---

## 10. Manufacturing constraints

- **FDM 3D printing:**
  - 0.4 mm nozzle, 220 × 220 × 250 mm bed. Split parts that don't fit, with keyed joints.
  - Minimum wall 2.4 mm (load-bearing at least 3.2 mm).
  - Ribs rather than solid sections.
  - Design for minimal supports, and state each part's print orientation.
  - Clearances: sliding fit 0.3 mm, press fit −0.1 mm. Holes for heat-set inserts per the insert maker's spec.
- **Window:** laser-cut 2–3 mm acrylic or polycarbonate. Give a 2D outline (DXF) with hole positions.
- **Off-the-shelf parts only beyond that:**
  - screws, inserts, foam and TPU gasket strip, thermal pad;
  - optional: 5 V fan, bubble vial.
- **Assembly:** standard hex keys and one hand-tight knob. No glue in the load path.

---

## 11. Notes for the operator (put these in the documentation)

These camera settings affect the pipeline as much as the enclosure does:
- Landscape, 1080p or 4K, **25 or 30 fps, constant frame rate**. Not HDR or Dolby Vision, unless the pipeline confirms it is supported.
- **Lock the lens.** Stop the phone switching between ultra-wide and wide automatically, for example via a pro video app.
- **Lock focus and exposure** after framing.
- **Stabilisation off or "standard",** not "cinematic" or "action". Strong stabilisation crops and warps the frame, which corrupts the camera model.
- Airplane mode on, screen off or dim, and the phone at room temperature before mounting.
- Before kick-off, check the battery, the power bank and that the phone isn't already warm.
- Avoid mounting on stand structures that bounce with the crowd. Pick the stiffest point available.
- Record which way up the phone is mounted (lens left or right), plus the mount height and the tilt marking.

---

## 12. What to deliver

1. **Parametric source** (CadQuery, OpenSCAD or FreeCAD), with all key parameters at the top:
   - the phone envelope parameters from section 4;
   - window gap, hood length, vent sizes, wall thickness;
   - mount type.
2. **STEP and STL per part**, plus the **DXF** of the window.
3. **An exploded assembly drawing**, and a section through the lens showing:
   - the field-of-view cone clearing the window and hood;
   - the lens gasket.
4. **A table of lens positions** relative to the mount datum for the small, standard and large phone classes, in each orientation.
5. **BOM** (hardware, inserts, gaskets, window) and **print settings** per part (material, orientation, layer height, walls, infill).
6. **Assumptions and open questions:** a short list of anything you had to assume, so it can be checked against real phones.

---

## 13. Acceptance tests (how the design will be judged)

| Test | Pass |
|---|---|
| Fit: smallest and largest supported phone, both orientations | Secure, no rattle, buttons not pressed, cable connects |
| Vignetting: record a white wall with the ultra-wide lens through the window | No dark corners and no visible enclosure edges anywhere in frame |
| Reflections: record into a bright scene, screen on full brightness | No ghosting or flare from the window or the inside |
| Thermal: 2.5 h of 4K30 recording in direct sun, 25–30 °C ambient, while charging | No thermal warning and no recording stop. Log phone temperature (target: phone back under 40 °C) |
| Stability: record a static checkerboard 10 m away while blowing a fan at about 10 m/s and tapping the mount | Image shake ≤ 2 px at 1080p. No drift after 2.5 h |
| Remount: remove and remount on the same tripod 5 times | View shifts ≤ 5 px |
| Rain: 10 minutes under a garden sprinkler | Phone dry, window clear, water drains |
| Drop: from 1.5 m onto grass with the phone fitted | Phone retained, no cracked load-bearing part |
| Pipeline: film 5 minutes of a real match and run it through the project's video pipeline | The pitch calibrates on most frames and the minimap tracks players |

---

## Units and conventions
- Millimetres unless stated otherwise.
- Angles in degrees.
- "Landscape lens-left" means: the phone held in landscape with the screen facing away from the pitch (the operator's side), and the camera module on the left as seen from behind the phone.
