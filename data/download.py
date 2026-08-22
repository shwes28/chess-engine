"""
data/download.py
================
Download chess games from Lichess API for training.

Lichess provides a free API to download games from any user.
We'll pull games from several top players to build a diverse dataset.

Usage:
    python data/download.py
    python data/download.py --games 2000 --output data/games.pgn
"""

import urllib.request
import urllib.error
import argparse
import time
import os
from pathlib import Path

# Top Lichess players with many high-quality games
TOP_PLAYERS = [
    "DrNykterstein",   # Magnus Carlsen
    "LyonBeast",       # Maxime Vachier-Lagrave
    "Hikaru",          # Hikaru Nakamura
    "nihalsarin2004",  # Nihal Sarin
    "FabianoCaruana",  # Fabiano Caruana
]

def download_games(username: str, output_file, max_games: int = 500) -> int:
    """
    Download games from a Lichess user via their API.
    Returns number of games downloaded.
    """
    url = (
        f"https://lichess.org/api/games/user/{username}"
        f"?max={max_games}&rated=true&perfType=blitz,rapid,classical"
        f"&clocks=false&evals=false&opening=false"
    )

    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": "ChessEngineTrainer/1.0"
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        print(f"  Downloading {max_games} games from {username}...")
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read().decode("utf-8", errors="ignore")
            game_count = data.count("[Event ")
            output_file.write(data)
            output_file.write("\n")
            print(f"  -> Got {game_count} games from {username}")
            return game_count
    except urllib.error.URLError as e:
        print(f"  -> Failed to download from {username}: {e}")
        return 0


def main(total_games: int = 1000, output_path: str = "data/games.pgn"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    games_per_player = max(100, total_games // len(TOP_PLAYERS))
    total_downloaded = 0

    print(f"\nDownloading ~{total_games} games from Lichess...")
    print(f"Output: {output_path}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        for player in TOP_PLAYERS:
            count = download_games(player, f, games_per_player)
            total_downloaded += count
            if total_downloaded >= total_games:
                break
            time.sleep(1)  # Be respectful to Lichess API rate limits

    print(f"\nDone! Total games downloaded: {total_downloaded}")
    size_kb = os.path.getsize(output_path) / 1024
    print(f"File size: {size_kb:.1f} KB")
    print(f"Saved to: {output_path}")
    return total_downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Lichess games")
    parser.add_argument("--games",  type=int, default=1000,           help="Total games to download")
    parser.add_argument("--output", type=str, default="data/games.pgn", help="Output PGN file path")
    args = parser.parse_args()
    main(args.games, args.output)
