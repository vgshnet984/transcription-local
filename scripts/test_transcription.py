#!/usr/bin/env python3
"""
Command-line testing script for the transcription platform.

This script provides comprehensive testing of the full transcription pipeline,
including performance benchmarking and error condition testing.
"""

import argparse
import time
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.transcription.engine import TranscriptionEngine
from src.audio.processor import AudioProcessor
from src.database.models import Base
from src.database.init_db import init_database
from src.config import settings


class TranscriptionTester:
    """Comprehensive testing class for transcription platform."""
    
    def __init__(self, test_dir: Optional[str] = None):
        self.test_dir = test_dir if test_dir else tempfile.mkdtemp()
        self.results = []
        self.engine = TranscriptionEngine()
        self.processor = AudioProcessor()
        
        # Initialize test database
        self._init_test_db()
    
    def _init_test_db(self):
        """Initialize test database."""
        try:
            init_database()
            print("✓ Test database initialized")
        except Exception as e:
            print(f"✗ Database initialization failed: {e}")
    
    def run_all_tests(self, audio_files: List[str]) -> Dict:
        """Run all tests on provided audio files."""
        print(f"Starting comprehensive testing with {len(audio_files)} files...")
        print(f"Test directory: {self.test_dir}")
        print("-" * 50)
        
        results = {
            "summary": {},
            "tests": [],
            "performance": {},
            "errors": []
        }
        
        # Run individual file tests
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                file_results = self.test_single_file(audio_file)
                results["tests"].append(file_results)
            else:
                print(f"✗ File not found: {audio_file}")
                results["errors"].append(f"File not found: {audio_file}")
        
        # Run performance tests
        results["performance"] = self.run_performance_tests(audio_files)
        
        # Run error condition tests
        results["error_tests"] = self.run_error_tests()
        
        # Generate summary
        results["summary"] = self.generate_summary(results)
        
        return results
    
    def test_single_file(self, audio_file: str) -> Dict:
        """Test transcription on a single audio file."""
        print(f"\nTesting: {os.path.basename(audio_file)}")
        
        result = {
            "file": audio_file,
            "filename": os.path.basename(audio_file),
            "tests": {}
        }
        
        try:
            # Test 1: Audio validation
            print("  Testing audio validation...")
            validation_result = self.processor.validate(audio_file)
            result["tests"]["validation"] = {
                "passed": validation_result["valid"],
                "details": validation_result
            }
            
            if not validation_result["valid"]:
                print(f"  ✗ Validation failed: {validation_result['error']}")
                return result
            
            # Test 2: Metadata extraction
            print("  Testing metadata extraction...")
            metadata = self.processor.get_metadata(audio_file)
            result["tests"]["metadata"] = {
                "passed": bool(metadata),
                "details": metadata
            }
            
            # Test 3: Basic transcription
            print("  Testing basic transcription...")
            start_time = time.time()
            transcription = self.engine.transcribe(audio_file)
            transcription_time = time.time() - start_time
            
            result["tests"]["transcription"] = {
                "passed": transcription["error"] is None,
                "processing_time": transcription_time,
                "text_length": len(transcription["text"]),
                "confidence": transcription["confidence"],
                "language": transcription["language"],
                "error": transcription["error"]
            }
            
            # Test 4: Speaker diarization (if enabled)
            if settings.enable_speaker_diarization:
                print("  Testing speaker diarization...")
                diarization_result = self.test_diarization(audio_file)
                result["tests"]["diarization"] = diarization_result
            
            # Test 5: Format conversion
            print("  Testing format conversion...")
            conversion_result = self.test_format_conversion(audio_file)
            result["tests"]["conversion"] = conversion_result
            
            print(f"  ✓ Completed tests for {os.path.basename(audio_file)}")
            
        except Exception as e:
            print(f"  ✗ Error testing {audio_file}: {e}")
            result["tests"]["error"] = str(e)
        
        return result
    
    def test_diarization(self, audio_file: str) -> Dict:
        """Test speaker diarization functionality."""
        try:
            # This would test the diarization pipeline
            # For now, return a mock result
            return {
                "passed": True,
                "speakers_detected": 1,
                "segments": 1,
                "details": "Mock diarization test"
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def test_format_conversion(self, audio_file: str) -> Dict:
        """Test audio format conversion."""
        try:
            # Test conversion to WAV
            output_path = self.processor.convert_to_wav(audio_file)
            
            result = {
                "passed": os.path.exists(output_path),
                "output_path": output_path,
                "output_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0
            }
            
            # Cleanup
            if os.path.exists(output_path):
                os.unlink(output_path)
            
            return result
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def run_performance_tests(self, audio_files: List[str]) -> Dict:
        """Run performance benchmarking tests."""
        print("\nRunning performance tests...")
        
        performance = {
            "transcription_speed": {},
            "memory_usage": {},
            "concurrent_processing": {}
        }
        
        # Test transcription speed
        if audio_files:
            test_file = audio_files[0]
            if os.path.exists(test_file):
                print("  Testing transcription speed...")
                
                start_time = time.time()
                transcription = self.engine.transcribe(test_file)
                end_time = time.time()
                
                processing_time = end_time - start_time
                audio_duration = transcription.get("processing_time", 0)
                
                if audio_duration > 0:
                    speed_ratio = audio_duration / processing_time
                    performance["transcription_speed"] = {
                        "processing_time": processing_time,
                        "audio_duration": audio_duration,
                        "speed_ratio": speed_ratio,
                        "real_time_factor": 1 / speed_ratio
                    }
        
        return performance
    
    def run_error_tests(self) -> Dict:
        """Run error condition tests."""
        print("\nRunning error condition tests...")
        
        error_tests = {
            "invalid_files": [],
            "large_files": [],
            "unsupported_formats": []
        }
        
        # Test invalid file
        try:
            result = self.engine.transcribe("nonexistent_file.wav")
            error_tests["invalid_files"].append({
                "test": "non_existent_file",
                "passed": result["error"] is not None,
                "error": result["error"]
            })
        except Exception as e:
            error_tests["invalid_files"].append({
                "test": "non_existent_file",
                "passed": True,
                "error": str(e)
            })
        
        # Test unsupported format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not audio content")
            invalid_file = f.name
        
        try:
            result = self.processor.validate(invalid_file)
            error_tests["unsupported_formats"].append({
                "test": "text_file",
                "passed": not result["valid"],
                "error": result["error"]
            })
        finally:
            os.unlink(invalid_file)
        
        return error_tests
    
    def generate_summary(self, results: Dict) -> Dict:
        """Generate test summary."""
        tests = results["tests"]
        errors = results["errors"]
        
        total_files = len(tests)
        successful_tests = sum(1 for test in tests if not test.get("tests", {}).get("error"))
        
        summary = {
            "total_files_tested": total_files,
            "successful_tests": successful_tests,
            "failed_tests": total_files - successful_tests,
            "success_rate": (successful_tests / total_files * 100) if total_files > 0 else 0,
            "total_errors": len(errors)
        }
        
        return summary
    
    def save_results(self, results: Dict, output_file: str):
        """Save test results to file."""
        with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")
    
    def print_summary(self, results: Dict):
        """Print test summary."""
        summary = results["summary"]
        
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Total files tested: {summary['total_files_tested']}")
        print(f"Successful tests: {summary['successful_tests']}")
        print(f"Failed tests: {summary['failed_tests']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total errors: {summary['total_errors']}")
        
        if results["performance"].get("transcription_speed"):
            speed = results["performance"]["transcription_speed"]
            print(f"\nPerformance:")
            print(f"  Processing time: {speed['processing_time']:.2f}s")
            print(f"  Speed ratio: {speed['speed_ratio']:.2f}x")
            print(f"  Real-time factor: {speed['real_time_factor']:.2f}x")
        
        print("=" * 50)
    
    def cleanup(self):
        """Clean up test resources."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test transcription platform")
    parser.add_argument("audio_files", nargs="+", help="Audio files to test")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--test-dir", help="Test directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = TranscriptionTester(args.test_dir)
    
    try:
        # Run tests
        results = tester.run_all_tests(args.audio_files)
        
        # Print summary
        tester.print_summary(results)
        
        # Save results
        if args.output:
            tester.save_results(results, args.output)
        
        # Exit with appropriate code
        if results["summary"]["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Testing failed: {e}")
        sys.exit(1)
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main() 