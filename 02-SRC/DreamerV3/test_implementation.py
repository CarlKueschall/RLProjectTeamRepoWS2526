"""
Test script to verify the simplified DreamerV3 implementation.

Run with:
    cd 02-SRC/DreamerV3
    python test_implementation.py

AI Usage Declaration:
This file was developed with assistance from Claude Code.
"""

import numpy as np
import torch


def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")

    from utils.math_ops import symlog, symexp, lambda_returns, ReturnNormalizer
    from utils.distributions import TanhNormal, GaussianDist
    from utils.buffer import EpisodeBuffer
    from models.sequence_model import SequenceModel
    from models.world_model import WorldModel
    from models.behavior import Behavior
    from agents.hockey_dreamer import HockeyDreamer

    print("  All imports successful!")
    return True


def test_math_ops():
    """Test mathematical operations."""
    print("Testing math_ops...")

    from utils.math_ops import symlog, symexp, lambda_returns

    # Test symlog/symexp
    x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    y = symlog(x)
    x_back = symexp(y)
    assert torch.allclose(x, x_back, atol=1e-5), "symlog/symexp roundtrip failed"

    # Test lambda_returns
    rewards = torch.rand(10, 32)
    values = torch.rand(10, 32)
    continues = torch.ones(10, 32)
    bootstrap = torch.rand(32)

    returns = lambda_returns(rewards, values, continues, bootstrap)
    assert returns.shape == (10, 32), f"Wrong shape: {returns.shape}"

    print("  math_ops OK!")
    return True


def test_distributions():
    """Test distribution classes."""
    print("Testing distributions...")

    from utils.distributions import TanhNormal, GaussianDist

    # Test TanhNormal
    mean = torch.zeros(32, 4)
    std = torch.ones(32, 4) * 0.5
    dist = TanhNormal(mean, std)

    sample = dist.sample()
    assert sample.shape == (32, 4), f"Wrong shape: {sample.shape}"
    assert (sample >= -1).all() and (sample <= 1).all(), "Sample out of bounds"

    log_prob = dist.log_prob(sample)
    assert log_prob.shape == (32,), f"Wrong log_prob shape: {log_prob.shape}"

    # Test GaussianDist
    g_dist = GaussianDist(mean, std)
    g_sample = g_dist.sample()
    assert g_sample.shape == (32, 4)

    print("  distributions OK!")
    return True


def test_buffer():
    """Test episode buffer."""
    print("Testing buffer...")

    from utils.buffer import EpisodeBuffer

    buffer = EpisodeBuffer(
        capacity=10000,
        obs_shape=(18,),
        action_shape=(4,),
    )

    # Add fake episodes
    for ep in range(5):
        for step in range(50):
            obs = np.random.randn(18).astype(np.float32)
            action = np.random.randn(4).astype(np.float32)
            reward = np.random.randn()
            done = step == 49
            is_first = step == 0
            buffer.add(obs, action, reward, done, is_first)

    # Sample batch
    batch = buffer.sample(batch_size=8, seq_length=20)
    assert batch is not None
    assert batch['obs'].shape == (8, 20, 18), f"Wrong obs shape: {batch['obs'].shape}"
    assert batch['action'].shape == (8, 20, 4), f"Wrong action shape: {batch['action'].shape}"

    print(f"  Buffer OK! ({len(buffer)} transitions, {buffer.num_episodes} episodes)")
    return True


def test_sequence_model():
    """Test RSSM sequence model."""
    print("Testing SequenceModel...")

    from models.sequence_model import SequenceModel

    model = SequenceModel(
        embed_dim=128,
        action_dim=4,
        hidden_size=64,
        latent_size=32,
        recurrent_size=64,
    )

    batch_size = 8
    seq_len = 20

    embeds = torch.randn(batch_size, seq_len, 128)
    actions = torch.randn(batch_size, seq_len, 4)
    is_first = torch.zeros(batch_size, seq_len)
    is_first[:, 0] = 1.0

    posteriors, priors = model.observe_sequence(embeds, actions, is_first)

    assert posteriors['h'].shape == (batch_size, seq_len, 64)
    assert posteriors['z'].shape == (batch_size, seq_len, 32)

    # Test imagination
    start_state = {k: v[:, -1] for k, v in posteriors.items()}

    # Simple policy for testing
    def dummy_policy(features):
        from utils.distributions import TanhNormal
        mean = torch.zeros(features.shape[0], 4)
        std = torch.ones(features.shape[0], 4) * 0.5
        return TanhNormal(mean, std)

    states, actions = model.imagine_sequence(dummy_policy, start_state, horizon=10)
    assert states['h'].shape == (batch_size, 10, 64)

    print("  SequenceModel OK!")
    return True


def test_world_model():
    """Test complete world model."""
    print("Testing WorldModel...")

    from models.world_model import WorldModel

    model = WorldModel(
        obs_dim=18,
        action_dim=4,
        hidden_size=64,
        latent_size=32,
        recurrent_size=64,
        embed_dim=64,
        device='cpu',
    )

    batch_size = 8
    seq_len = 20

    batch = {
        'obs': torch.randn(batch_size, seq_len, 18),
        'action': torch.randn(batch_size, seq_len, 4),
        'reward': torch.randn(batch_size, seq_len),
        'is_first': torch.zeros(batch_size, seq_len),
        'is_terminal': torch.zeros(batch_size, seq_len),
    }
    batch['is_first'][:, 0] = 1.0

    loss, metrics = model.compute_loss(batch)
    assert loss.dim() == 0, "Loss should be scalar"
    assert 'world/loss' in metrics

    print(f"  WorldModel OK! (loss={loss.item():.4f})")
    return True


def test_behavior():
    """Test actor-critic behavior model."""
    print("Testing Behavior...")

    from models.behavior import Behavior

    behavior = Behavior(
        feature_dim=96,  # 64 + 32
        action_dim=4,
        hidden_size=64,
        device='cpu',
    )

    batch_size = 8
    horizon = 15

    states = {
        'h': torch.randn(batch_size, horizon, 64),
        'z': torch.randn(batch_size, horizon, 32),
    }
    actions = torch.randn(batch_size, horizon, 4)
    rewards = torch.randn(batch_size, horizon)
    continues = torch.ones(batch_size, horizon)

    actor_loss, critic_loss, metrics = behavior.train_step(
        states, actions, rewards, continues
    )

    assert actor_loss.dim() == 0
    assert critic_loss.dim() == 0
    assert 'behavior/actor_loss' in metrics

    print(f"  Behavior OK! (actor_loss={actor_loss.item():.4f})")
    return True


def test_hockey_dreamer():
    """Test complete HockeyDreamer agent."""
    print("Testing HockeyDreamer...")

    from agents.hockey_dreamer import HockeyDreamer

    agent = HockeyDreamer(
        obs_dim=18,
        action_dim=4,
        hidden_size=64,
        num_categories=8,  # Small for testing
        num_classes=8,     # Small for testing
        recurrent_size=64,
        embed_dim=64,
        horizon=10,
        device='cpu',
    )

    # Test acting
    obs = np.random.randn(18).astype(np.float32)
    action = agent.act(obs)
    assert action.shape == (4,), f"Wrong action shape: {action.shape}"
    assert (action >= -1).all() and (action <= 1).all(), "Action out of bounds"

    # Test training
    batch = {
        'obs': np.random.randn(8, 20, 18).astype(np.float32),
        'action': np.random.randn(8, 20, 4).astype(np.float32),
        'reward': np.random.randn(8, 20).astype(np.float32),
        'is_first': np.zeros((8, 20), dtype=np.float32),
        'is_terminal': np.zeros((8, 20), dtype=np.float32),
    }
    batch['is_first'][:, 0] = 1.0

    metrics = agent.train_step(batch)
    assert 'world/loss' in metrics
    assert 'behavior/actor_loss' in metrics

    # Test state save/restore
    state = agent.state()
    agent.restore_state(state)

    print("  HockeyDreamer OK!")
    return True


def test_with_environment():
    """Test with actual hockey environment."""
    print("Testing with HockeyEnv...")

    try:
        from envs.hockey_wrapper import HockeyEnvDreamer
        from agents.hockey_dreamer import HockeyDreamer
        from utils.buffer import EpisodeBuffer

        # Create environment
        env = HockeyEnvDreamer(mode='NORMAL', opponent='weak')

        # Create agent
        agent = HockeyDreamer(
            obs_dim=18,
            action_dim=4,
            hidden_size=64,
            num_categories=8,  # Small for testing
            num_classes=8,     # Small for testing
            recurrent_size=64,
            horizon=10,
            device='cpu',
        )

        # Create buffer
        buffer = EpisodeBuffer(
            capacity=10000,
            obs_shape=(18,),
            action_shape=(4,),
        )

        # Run a few episodes
        for ep in range(3):
            obs, _ = env.reset()
            agent.reset()
            done = False
            is_first = True
            total_reward = 0

            while not done:
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.add(obs, action, reward, done, is_first)
                is_first = False

                obs = next_obs
                total_reward += reward

            print(f"    Episode {ep+1}: reward={total_reward:.2f}, winner={info.get('winner', 0)}")

        # Train on collected data
        if len(buffer) > 0:
            batch = buffer.sample(batch_size=4, seq_length=20)
            if batch is not None:
                metrics = agent.train_step(batch)
                print(f"    Training: world_loss={metrics['world/loss']:.4f}")

        env.close()
        print("  Environment test OK!")
        return True

    except Exception as e:
        print(f"  Environment test FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("DreamerV3 Implementation Tests")
    print("=" * 60)
    print()

    tests = [
        ("Imports", test_imports),
        ("Math Operations", test_math_ops),
        ("Distributions", test_distributions),
        ("Episode Buffer", test_buffer),
        ("Sequence Model", test_sequence_model),
        ("World Model", test_world_model),
        ("Behavior Model", test_behavior),
        ("HockeyDreamer Agent", test_hockey_dreamer),
        ("Hockey Environment", test_with_environment),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nAll tests passed! Ready for training.")
        print("\nTo train, run:")
        print("  python train_hockey.py --max_steps 10000 --no_wandb")
    else:
        print("\nSome tests failed. Please check the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
