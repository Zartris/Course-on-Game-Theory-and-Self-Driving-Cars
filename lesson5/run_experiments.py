#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run multiple auction and bidding strategy comparison experiments.
"""

import subprocess
import os
import time
import argparse

def ensure_directory_exists(directory):
    """Ensure a directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def run_auction_comparisons(output_dir="results", verbose=False):
    """Run auction mechanism comparisons with different parameters."""
    print("\n===== Running Auction Mechanism Comparisons =====")
    
    # Configurations to test
    configs = [
        # Format: (num_robots, num_tasks, max_steps, name)
        (5, 15, 80, "small"),
        (8, 25, 120, "medium"),
        (12, 40, 160, "large")
    ]
    
    for num_robots, num_tasks, max_steps, size in configs:
        print(f"\nRunning {size} configuration:")
        print(f"  Robots: {num_robots}, Tasks: {num_tasks}, Steps: {max_steps}")
        
        # Create specific output directory for this run
        run_output_dir = os.path.join(output_dir, f"auction_comparison_{size}")
        ensure_directory_exists(run_output_dir)
        
        # Build command
        cmd = [
            "python", "auction_comparison.py",
            "--num_robots", str(num_robots),
            "--num_tasks", str(num_tasks),
            "--max_steps", str(max_steps),
            "--output_dir", run_output_dir
        ]
        
        # Run command
        try:
            if verbose:
                # Show output in real-time
                subprocess.run(cmd, check=True)
            else:
                # Capture output
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                # Print summary
                print(f"  Completed {size} auction comparison")
        except subprocess.CalledProcessError as e:
            print(f"Error running {size} auction comparison: {e}")
            if e.output:
                print(f"Output: {e.output}")
            if e.stderr:
                print(f"Error: {e.stderr}")

def run_bidding_strategy_comparisons(output_dir="results", verbose=False):
    """Run bidding strategy comparisons with different auction mechanisms."""
    print("\n===== Running Bidding Strategy Comparisons =====")
    
    # Auction types to test
    auction_types = ["sequential", "parallel", "combinatorial"]
    payment_rules = ["first_price", "second_price"]
    
    # Skip invalid combinations
    skip_combinations = [
        ("combinatorial", "second_price")
    ]
    
    for auction_type in auction_types:
        for payment_rule in payment_rules:
            # Skip invalid combinations
            if (auction_type, payment_rule) in skip_combinations:
                print(f"Skipping {auction_type} with {payment_rule} (incompatible)")
                continue
                
            print(f"\nRunning bidding comparison with {auction_type} auction and {payment_rule} payment:")
            
            # Create specific output directory for this run
            run_output_dir = os.path.join(output_dir, f"bidding_comparison_{auction_type}_{payment_rule}")
            ensure_directory_exists(run_output_dir)
            
            # Build command
            cmd = [
                "python", "bidding_strategy_comparison.py",
                "--auction_type", auction_type,
                "--payment_rule", payment_rule,
                "--robots_per_strategy", "2",
                "--num_tasks", "20",
                "--max_steps", "100",
                "--output_dir", run_output_dir
            ]
            
            # Run command
            try:
                if verbose:
                    # Show output in real-time
                    subprocess.run(cmd, check=True)
                else:
                    # Capture output
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    # Print summary
                    print(f"  Completed bidding comparison for {auction_type}_{payment_rule}")
            except subprocess.CalledProcessError as e:
                print(f"Error running bidding comparison for {auction_type}_{payment_rule}: {e}")
                if e.output:
                    print(f"Output: {e.output}")
                if e.stderr:
                    print(f"Error: {e.stderr}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run auction and bidding experiments")
    
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save output files")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output during execution")
    parser.add_argument("--auction_only", action="store_true",
                        help="Run only auction mechanism comparisons")
    parser.add_argument("--bidding_only", action="store_true",
                        help="Run only bidding strategy comparisons")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Ensure output directory exists
    ensure_directory_exists(args.output_dir)
    
    # Record start time
    start_time = time.time()
    
    # Run experiments based on flags
    if args.bidding_only:
        run_bidding_strategy_comparisons(args.output_dir, args.verbose)
    elif args.auction_only:
        run_auction_comparisons(args.output_dir, args.verbose)
    else:
        # Run both by default
        run_auction_comparisons(args.output_dir, args.verbose)
        run_bidding_strategy_comparisons(args.output_dir, args.verbose)
    
    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()