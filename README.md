Pi Searcher

This project is made for searching through a given binary sequence for a specified binary string. Specifically, frames of a video are searched using GPU parallelization with CUDA. It also includes tools to convert videos to the given format and back.

How to Run This Yourself

This project is a mix of AI-generated code and my own bad code. I cannot guarantee that it will work on any other machine.

Requirements

CUDA 12 and a compatible graphics card

Lots of patience to get this running, as well as to handle the sheer number of calculations required

Considerations Before Running

Even though this code is somewhat optimized, the nature of the Hamming distance algorithm (used to find the closest match) means it takes at least:

(length of sequence) × 0.5 × (length of searched string) × (number of frames)

comparisons to find the closest matches. That means it can take billions and billions of calculations before reaching a result. However, the quality of the output doesn’t scale linearly, so you’ll have to find the best ratio between quality and runtime yourself.

When choosing resolution and sequence length, consider the following:

Resolution: This has the greatest impact on the quality and recognizability of the output. The default value is 8x6, which is best for a sequence length of 10 billion bits.

Sequence length: This has the biggest impact on runtime and scales linearly. Also, the longer it is, the better the Hamming distance. I recommend at least a few billion bits to get anything recognizable.

Rough Step-by-Step

1. Generate Sequence

First, generate your sequence as a text file. Then, use "sequenceconversion.py" to encode it to binary:

Even numbers = 0

Odd numbers = 1

Or modify the script to fit your encoding needs. This will generate a NumPy file that you will use as the input.

2. Prepare Frames

Next, choose a resolution for the frames of your video. In every file with img_height and img_width (or something similar), change it to your desired output. Calculate the bit length as:

width × height

In "search.py," change every reference to the old bit length in the file to your new one (I didn’t find any way to automate this). Then, convert your video to another NumPy file using "videoconversion.py." This will also generate reference frames of just the original video.

3. Perform the Search

Change the block size if needed to get the best performance for your GPU. Then, just run the file and wait for it to finish.

4. Generate Output

If you used a video, just run "outputconversion.py" to generate a video again based on the output, and you're done!

Final Thoughts

I hope you have fun with it. I know it’s not much, but I’m absolutely happy about any forks and videos made for this project. If you have any questions, just contact me on Discord: @fopetix.

