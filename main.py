"""Entry point for the satellite orbit visualiser."""

from satellite_sim import SimulationApp


def main() -> None:
    app = SimulationApp()
    app.run()


if __name__ == "__main__":
    main()
