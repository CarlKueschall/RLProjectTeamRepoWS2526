import os
import tempfile

import numpy as np

def create_gif_for_wandb(env, agent, opponent, mode, max_timesteps, num_episodes=3, eps=0.0,
                         self_play_opponent=None):
    ######################################################
    # Record multiple episodes and stitch them horizontally into a single GIF.
    
    #Arguments:
    #env: Hockey environment
    #agent: TD3 agent
    # opponent: Opponent agent
    # mode: Game mode
    # max_timesteps: Max steps per episode
    # num_episodes: Number of episodes to record
    # eps: Exploration epsilon
    # self_play_opponent: Self-play opponent network (if in self-play stage)
    #Returns:
    # gif_frames: List of horizontally stitched RGB frames
    # results: List of episode results (win/loss/tie)
    from .frame_capture import record_episode_frames

    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. GIF recording disabled. Install with: pip install Pillow")
        return None, []  # can't make gifs without PIL

    all_episode_frames = []
    results = []

    #########################################################
    # Record each episode
    for _ in range(num_episodes):
        frames, winner = record_episode_frames(
            env, agent, opponent, mode, max_timesteps, eps,
            self_play_opponent=self_play_opponent
        )
        all_episode_frames.append(frames)
        results.append(winner)

    if not all_episode_frames or not all_episode_frames[0]:
        return None, results  # no frames to work with

    #########################################################
    # Find the max number of frames across all episodes
    max_frames = 0
    for frames in all_episode_frames:
        frame_count = len(frames)
        if frame_count > max_frames:
            max_frames = frame_count

    #########################################################
    # Pad shorter episodes by repeating last frame
    #########################################################
    for frames in all_episode_frames:
        if len(frames) < max_frames and len(frames) > 0:
            last_frame = frames[-1]
            while len(frames) < max_frames:
                frames.append(last_frame)  # pad with last frame so all episodes same length

    #########################################################
    # Stitch frames horizontally
    stitched_frames = []
    for frame_idx in range(max_frames):
        episode_frames_at_idx = []
        for ep_frames in all_episode_frames:
            if frame_idx < len(ep_frames):
                episode_frames_at_idx.append(ep_frames[frame_idx])

        if episode_frames_at_idx:
            #########################################################
            # Convert to PIL images and stitch horizontally
            pil_images = []
            for f in episode_frames_at_idx:
                pil_img = Image.fromarray(f)
                pil_images.append(pil_img)

            # Calculate total width and max height
            total_width = 0
            for img in pil_images:
                total_width += img.width  # add up all widths
            
            max_height = 0
            for img in pil_images:
                img_height = img.height
                if img_height > max_height:
                    max_height = img_height  # take tallest one

            #########################################################
            # Create stitched image
            #########################################################
            stitched = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in pil_images:
                stitched.paste(img, (x_offset, 0))
                x_offset += img.width  # move over for next image

            stitched_frames.append(np.array(stitched))
        #########################################################

    return stitched_frames, results


#########################################################
#########################################################
def save_gif_to_wandb(frames, results, episode_num, run_name, metric_name="behavior/gameplay_gif"):
    #########################################################
    # Save GIF frames to W&B.
    #Arguments:
    # frames: List of RGB frames (stitched)
    # results: List of episode results
    # episode_num: Current episode number
    # run_name: W&B run name
    # metric_name: W&B metric name (e.g., "behavior/gameplay_gif_vs_target")
    if frames is None or len(frames) == 0:
        return

    try:
        import imageio
        import wandb
    except ImportError as e:
        print(f"GIF recording failed - missing dependency: {e}")
        return

    try:
        #########################################################
        # Create temp file for GIF
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            tmp_path = tmp.name

        # Save as GIF (15 fps for smooth playback)
        imageio.mimsave(tmp_path, frames, fps=15, loop=0)

        #########################################################
        # Create caption with results
        result_strs = []
        for i, r in enumerate(results):
            if r == 1:
                result_strs.append(f"Ep{i+1}:WIN")
            elif r == -1:
                result_strs.append(f"Ep{i+1}:LOSS")
            else:
                result_strs.append(f"Ep{i+1}:TIE")  # tie or timeout
        caption = f"Episode {episode_num} | {' | '.join(result_strs)}"

        #########################################################
        # Log to W&B
        #########################################################
        wandb.log({
            metric_name: wandb.Video(tmp_path, fps=15, format="gif", caption=caption)
        })

        # Clean up temp file
        os.unlink(tmp_path)  # delete temp file after uploading

        print(f"GIF recorded at episode {episode_num}: {', '.join(result_strs)}")

    except Exception as e:
        print(f"GIF recording failed: {e}")
#########################################################