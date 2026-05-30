from typing import List

def compute_fibonacci(n: int) -> List[int]:
    """Compute Fibonacci sequence up to index n using dynamic programming.
    
    Args:
        n: The highest index to compute (0-based).
        
    Returns:
        List of Fibonacci numbers from index 0 to n.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [0]
    if n == 1:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def generate_fibonacci_report(sequence: List[int]):
    """Generate a CSV report of the Fibonacci sequence.
    
    Args:
        sequence: List of Fibonacci numbers from index 0 to n.
    """
    with open('fibonacci_report.csv', 'w') as f:
        f.write("N,Fibonacci\n")
        for i, value in enumerate(sequence):
            f.write(f"{i},{value}\n")

def main():
    """Execute the Fibonacci computation pipeline."""
    # Compute up to the 20th index (0-based)
    fib_sequence = compute_fibonacci(20)
    
    # Test edge cases
    assert fib_sequence[0] == 0, "Fibonacci(0) should be 0"
    assert fib_sequence[1] == 1, "Fibonacci(1) should be 1"
    
    # Print textual representation
    print("Dynamic Programming Fibonacci Sequence (0 to 20):")
    for i, value in enumerate(fib_sequence):
        print(f"Step {i}: {value}")
    
    # Generate CSV report
    generate_fibonacci_report(fib_sequence)

if __name__ == "__main__":
    main()
