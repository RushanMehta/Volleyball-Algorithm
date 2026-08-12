(function () {
    "use strict";

    // Reliable browser download using Blob instead of a data URL.
    window.exportProfilesJSON = function () {
        try {
            const payload = {
                version: window.SCHEMA_VERSION || 6,
                exportedAt: new Date().toISOString(),
                players: typeof window.getRoster === "function"
                    ? window.getRoster()
                    : {}
            };

            const blob = new Blob(
                [JSON.stringify(payload, null, 2)],
                { type: "application/json;charset=utf-8" }
            );

            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");

            link.href = url;
            link.download = "coaches_volleyball_profiles.json";
            link.style.display = "none";

            document.body.appendChild(link);
            link.click();
            link.remove();

            setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch (error) {
            alert("Export failed: " + error.message);
            console.error("Profile export error:", error);
        }
    };

    // Accepts UTF-8 BOMs, spaces, underscores, hyphens, and parentheses.
    window.normalizeCSVHeader = function (value) {
        return String(value || "")
            .replace(/^\uFEFF/, "")
            .trim()
            .toLowerCase()
            .replace(/\([^)]*\)/g, "")
            .replace(/[_-]+/g, " ")
            .replace(/\s+/g, " ");
    };

    // Improve compatibility with GameChanger and other stat exports.
    if (window.CSV_HEADER_ALIASES) {
        window.CSV_HEADER_ALIASES.player = [
            "player",
            "name",
            "player name",
            "athlete",
            "athlete name"
        ];

        window.CSV_HEADER_ALIASES.total = [
            "sa",
            "serve att",
            "serve attempts",
            "serve attempt",
            "serves",
            "total serves",
            "attempts",
            "s att",
            "srv att",
            "service attempts"
        ];

        window.CSV_HEADER_ALIASES.ace = [
            "ace",
            "aces",
            "sa ace",
            "service aces",
            "service ace"
        ];

        window.CSV_HEADER_ALIASES.error = [
            "se",
            "serve err",
            "serve errors",
            "errors",
            "service errors",
            "service error",
            "err"
        ];
    }

    // Make Import accept older exports and validate each profile safely.
    window.importProfilesJSON = function (event) {
        const file = event.target.files && event.target.files[0];
        if (!file) return;

        const reader = new FileReader();

        reader.onload = function (loadEvent) {
            try {
                const parsed = JSON.parse(loadEvent.target.result);
                const players = parsed.players || parsed.roster;

                if (!players || typeof players !== "object") {
                    throw new Error("The file does not contain volleyball profiles.");
                }

                const cleanedPlayers = {};

                Object.entries(players).forEach(([name, profile]) => {
                    if (
                        profile &&
                        Array.isArray(profile.ts) &&
                        Array.isArray(profile.fl) &&
                        profile.ts.length === 6 &&
                        profile.fl.length === 6
                    ) {
                        cleanedPlayers[String(name)] = {
                            ts: profile.ts.map(Number),
                            fl: profile.fl.map(Number)
                        };
                    }
                });

                if (Object.keys(cleanedPlayers).length === 0) {
                    throw new Error("No valid player profiles were found.");
                }

                if (typeof window.saveRoster !== "function") {
                    throw new Error("Roster storage is unavailable.");
                }

                window.saveRoster(cleanedPlayers);
                alert(
                    `Loaded ${Object.keys(cleanedPlayers).length} player profile` +
                    `${Object.keys(cleanedPlayers).length === 1 ? "" : "s"}.`
                );
            } catch (error) {
                alert("Import failed: " + error.message);
                console.error("Profile import error:", error);
            } finally {
                event.target.value = "";
            }
        };

        reader.onerror = function () {
            alert("The selected file could not be read.");
        };

        reader.readAsText(file);
    };
})();
