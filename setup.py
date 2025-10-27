from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
from pathlib import Path
import subprocess
import sys
import os

def run_post_install():
    """Run all post-installation steps."""
    print("\n" + "="*60)
    print("🚀 Boltz Design Post-Installation")
    print("="*60 + "\n")
    
    # Step 1: Install PyRosetta
    print("⏳ [1/6] Installing PyRosetta (this may take a while)...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            'pyrosettacolabsetup', 'pyrosetta-installer', '--quiet'
        ])
        
        # Run PyRosetta installer
        import pyrosetta_installer
        pyrosetta_installer.install_pyrosetta()
        print("✅ PyRosetta installed\n")
    except Exception as e:
        print(f"⚠️  PyRosetta installation failed: {e}")
        print("   Continuing with other steps...\n")
    
    # Step 2: Fix NumPy/Numba compatibility (after PyRosetta)
    print("🩹 [2/6] Fixing NumPy/Numba compatibility...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '--upgrade', 'numpy>=1.24,<1.27', 'numba', '--quiet'
        ])
        print("✅ NumPy/Numba updated\n")
    except Exception as e:
        print(f"⚠️  Warning: {e}\n")
    
    # Step 3: Download Boltz weights
    print("⬇️  [3/6] Downloading Boltz weights and dependencies...")
    try:
        from boltz.main import download_boltz2
        cache = Path('~/.boltz').expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache)
        print('✅ Boltz weights downloaded successfully!\n')
    except Exception as e:
        print(f"⚠️  Could not auto-download: {e}")
        print("   Run this manually after installation:")
        print("   python -c 'from boltz.main import download_boltz2; from pathlib import Path; download_boltz2(Path(\"~/.boltz\").expanduser())'\n")
    
    # Step 4: Setup LigandMPNN
    print("🧬 [4/6] Setting up LigandMPNN...")
    try:
        # Find LigandMPNN directory - check multiple locations
        ligandmpnn_dir = None
        
        # Check current directory first
        if Path('LigandMPNN').exists():
            ligandmpnn_dir = Path('LigandMPNN')
        
        # Check in site-packages
        if not ligandmpnn_dir:
            for site_packages in sys.path:
                potential_path = Path(site_packages) / 'LigandMPNN'
                if potential_path.exists() and (potential_path / 'get_model_params.sh').exists():
                    ligandmpnn_dir = potential_path
                    break
        
        if ligandmpnn_dir and ligandmpnn_dir.exists():
            model_params_dir = ligandmpnn_dir / 'model_params'
            model_params_dir.mkdir(exist_ok=True)
            get_params_script = ligandmpnn_dir / 'get_model_params.sh'
            
            if get_params_script.exists():
                # Make script executable and run it
                os.chmod(get_params_script, 0o755)
                result = subprocess.run(
                    ['bash', str(get_params_script), str(model_params_dir)],
                    cwd=str(ligandmpnn_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("✅ LigandMPNN model parameters downloaded\n")
                else:
                    print(f"⚠️  LigandMPNN setup had issues: {result.stderr}\n")
            else:
                print("⚠️  get_model_params.sh not found\n")
        else:
            print("⚠️  LigandMPNN directory not found, skipping...\n")
    except Exception as e:
        print(f"⚠️  LigandMPNN setup failed: {e}")
        print("   You may need to run manually:\n")
        print("   cd LigandMPNN && bash get_model_params.sh './model_params'\n")
    
    # Step 5: Make DAlphaBall executable
    print("🔧 [5/6] Setting up DAlphaBall...")
    try:
        # Try multiple possible locations
        possible_paths = []
        
        # Check current directory
        if Path('boltz/utils/DAlphaBall.gcc').exists():
            possible_paths.append(Path('boltz/utils/DAlphaBall.gcc'))
        
        # Add site-packages paths
        for site_dir in sys.path:
            site_path = Path(site_dir)
            if site_path.exists() and 'site-packages' in str(site_path):
                possible_paths.append(site_path / "boltz" / "utils" / "DAlphaBall.gcc")
        
        # Add direct paths
        if sys.prefix:
            possible_paths.append(
                Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / 
                "site-packages" / "boltz" / "utils" / "DAlphaBall.gcc"
            )
        
        found = False
        for dalphaball in possible_paths:
            if dalphaball.exists():
                os.chmod(dalphaball, 0o755)
                print(f"✅ DAlphaBall.gcc set as executable at {dalphaball}\n")
                found = True
                break
        
        if not found:
            print("⚠️  DAlphaBall.gcc not found in expected locations\n")
            print("   You may need to chmod it manually after installation\n")
    except Exception as e:
        print(f"⚠️  Warning: {e}\n")
    
    # Step 6: Setup Jupyter kernel
    print("📓 [6/6] Setting up Jupyter kernel...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'ipykernel', 'install',
            '--user', '--name=boltz_ph',
            '--display-name=Boltz Protein Hunter'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Jupyter kernel 'Boltz Protein Hunter' installed\n")
    except Exception as e:
        print(f"⚠️  Jupyter kernel setup skipped (install ipykernel if needed)\n")
    
    # Final message
    print("="*60)
    print("🎉 Installation Complete!")
    print("="*60)
    print("\n✅ All components installed successfully!")
    print("\n🚀 Quick Start:")
    print("   from boltz.main import Boltz")
    print("   model = Boltz()")
    print("\n📓 Jupyter: Select 'Boltz Protein Hunter' kernel")
    print("="*60 + "\n")


class PostInstallCommand(install):
    """Custom post-installation steps."""
    def run(self):
        # Run standard install first
        install.run(self)
        # Then run our custom steps
        run_post_install()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # Run standard develop first
        develop.run(self)
        # Then run our custom steps
        run_post_install()


# Main setup configuration
setup(
    name='boltz-design',
    version='0.1.0',
    description='Boltz protein structure prediction and design environment',
    long_description=open('README.md').read() if Path('README.md').exists() else 'Boltz protein structure prediction and design environment',
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/boltz-design',
    
    # Package discovery - find ALL packages, don't restrict
    packages=find_packages(where='.', exclude=['tests*', 'docs*', 'examples*']),
    include_package_data=True,
    
    # Python version requirement
    python_requires='>=3.10,<3.11',
    
    # Core dependencies installed BEFORE post-install runs
    install_requires=[
        # Visualization
        'matplotlib',
        'seaborn',
        # Protein structure handling
        'prody',
        'pypdb',
        'py3Dmol',
        # Utilities
        'tqdm',
        'PyYAML',
        'requests',
        'logmd==0.1.45',
        'ml_collections',
        # Numerical (will be upgraded in post-install)
        'numpy>=1.23',
        'numba',
        # Jupyter
        'ipykernel',
    ],
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'black',
            'flake8',
            'ipython',
        ],
    },
    
    # Hook post-install commands
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    
    # Include binary files and scripts - use MANIFEST.in instead
    package_data={
        '': ['*.gcc', '*.sh', '*.json', '*.yaml', '*.yml'],
    },
    
    # Don't create zip file (needed for binary executable)
    zip_safe=False,
    
    # PyPI classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
    ],
)