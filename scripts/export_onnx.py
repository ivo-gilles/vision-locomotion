#!/usr/bin/env python3
"""
Export trained policy to ONNX for deployment.
Updated for Windows compatibility.
"""
import argparse
import torch
import torch.onnx
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from policies.monolithic import MonolithicPolicy
from policies.rma import RMAPolicy


def export_to_onnx(policy, output_path, phase=2, device='cuda:0'):
    """Export policy to ONNX format."""
    policy.eval()
    policy = policy.to(device)
    
    # Create dummy inputs
    batch_size = 1
    
    if phase == 1:
        # Phase 1: scandots
        dummy_proprio = torch.randn(batch_size, 48, device=device)
        dummy_scandots = torch.randn(batch_size, 16, 3, device=device)
        dummy_commands = torch.randn(batch_size, 3, device=device)
        dummy_inputs = {
            'proprioception': dummy_proprio,
            'scandots': dummy_scandots,
            'commands': dummy_commands
        }
    else:
        # Phase 2: depth
        dummy_proprio = torch.randn(batch_size, 48, device=device)
        dummy_depth = torch.randn(batch_size, 58, 87, device=device)
        dummy_commands = torch.randn(batch_size, 3, device=device)
        dummy_inputs = {
            'proprioception': dummy_proprio,
            'depth': dummy_depth,
            'commands': dummy_commands
        }
    
    # Export
    print(f"Exporting policy to {output_path}...")
    
    # Note: ONNX export for recurrent models is complex
    # This is a simplified version - may need adjustments
    try:
        torch.onnx.export(
            policy,
            dummy_inputs,
            output_path,
            input_names=list(dummy_inputs.keys()),
            output_names=['actions', 'values'],
            dynamic_axes={
                'proprioception': {0: 'batch_size'},
                'scandots' if phase == 1 else 'depth': {0: 'batch_size'},
                'commands': {0: 'batch_size'},
                'actions': {0: 'batch_size'},
                'values': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True
        )
        print("Export successful!")
    except Exception as e:
        print(f"Export failed: {e}")
        print("Note: ONNX export for recurrent models may require special handling.")


def main():
    parser = argparse.ArgumentParser(description='Export policy to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--phase', type=int, default=2, help='Phase (1 or 2)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    
    args = parser.parse_args()
    
    # Load checkpoint (Windows-compatible paths)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / args.checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=args.device)
    config = checkpoint.get('config', {})
    
    # Create policy
    architecture = config.get('policy', {}).get('architecture', 'monolithic')
    
    if architecture == 'rma':
        policy = RMAPolicy(config['policy'], phase=args.phase)
    else:
        policy = MonolithicPolicy(config['policy'], phase=args.phase)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Export (Windows-compatible paths)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_to_onnx(policy, str(output_path), args.phase, args.device)


if __name__ == '__main__':
    main()
