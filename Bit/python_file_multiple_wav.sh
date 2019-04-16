DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"
for filename in *.wav ; do
    python3 ../../Bit/binary_test.py "$(filename.wav)"
done