# ffx2_auto_tower_calibration
A Python script that automatically calibrates lightning towers in Final Fantasy X-2 using image recognition. This script supports all tower calibration methods including Rikku, Yuna, Paine, and Yuna's reverse tower (tower 10).

## How It Works

The script uses computer vision to:
1. Capture real-time screenshots of the game
2. Detect button prompts (↑, ↓, ←, →, PgUp, PgDn)
3. Process different calibration patterns:
   - Rikku: Detects and handles multiple simultaneous button presses
   - Yuna: Records and replays button sequences in order
   - Paine: Tracks buttons and watches for sparkle affect to determine when to press the button
   - Yuna Reverse: Records and replays sequences in reverse order (for tower 10)

## Requirements

- Python 3.x
- Required packages: OpenCV, NumPy, PyAutoGUI, keyboard
- Only tested running the game at 1920x1080
- Game must be in windowed fullscreen mode

## Usage

1. Run the script using Python
2. Select your calibration method when prompted:
   - 1: Rikku tower calibration
   - 2: Yuna tower calibration
   - 3: Paine tower calibration
   - 4: Yuna reverse tower calibration (tower 10)
3. Position your character at the appropriate tower

## Some Notes

- Its not perfect, but it works for the most part, you might have a few fails, but you can just run it again.
- I only tested at max graphics settings, so it might not work at lower settings.
- I only tested at 1920x1080, so it might not work at other resolutions.
- It only has recognition for keyboard inputs, so if you use a controller, you will need to use the keyboard to input the buttons.
- It is very badly written, but it works... there is alot of repeated code, and it is not optimized at all as i did each implementation as i went along and just added it to the script trying different things to get it to work... but it works.

## Troubleshooting

If detection issues occur:
- Verify game is in 1920x1080 resolution
- Check that template images in the "images" folder match your game's button prompts
- Ensure stable framerate for consistent timing
- Ensure the game window is clearly visible
- Check that no other windows are overlapping the game
- Ensure all required libraries are installed

## Known Limitations

- Currently optimized for 1920x1080 resolution
- Works best in windowed mode
- Performance depends on system capabilities, slower systems may have issues with recognition due to the amount of processing power required to run the recognition/screenshot capture. I cannot confirm it actually would fail on lower end systems, but it may.

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.
