"""Integration tests for the /fonts routes registered via the media router.

These cover the bug where the previous handler swallowed HTTPException and
turned auth failures into a 500, which surfaced in the frontend as
``Failed to load fonts (500)`` from ``Home.refreshFonts``.
"""

import pytest


@pytest.mark.asyncio
async def test_fonts_endpoint_returns_401_without_signed_auth(client):
    response = await client.get("/fonts")

    assert response.status_code == 401
    payload = response.json()
    assert "detail" in payload


@pytest.mark.asyncio
async def test_fonts_endpoint_returns_401_for_unsigned_request_when_disabled(
    client, app
):
    app.state.config.allow_unsigned_backend_auth = False

    response = await client.get(
        "/fonts",
        headers={"x-supoclip-user-id": "user-1"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_fonts_endpoint_returns_200_for_signed_request(client, auth_headers):
    response = await client.get("/fonts", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert "fonts" in payload
    assert isinstance(payload["fonts"], list)


@pytest.mark.asyncio
async def test_fonts_endpoint_does_not_swallow_http_exception(client):
    # Hit the endpoint with completely invalid auth headers. The previous
    # implementation caught HTTPException and re-raised it as 500, masking
    # the real status code (and the real reason) from the frontend.
    response = await client.get(
        "/fonts",
        headers={
            "x-supoclip-user-id": "user-1",
            "x-supoclip-ts": "not-a-number",
            "x-supoclip-signature": "deadbeef",
        },
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_font_file_returns_404_for_missing_font(client, auth_headers):
    response = await client.get("/fonts/this-font-does-not-exist", headers=auth_headers)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_font_file_serves_existing_system_font(client, auth_headers):
    # The first available bundled .ttf should be servable. We don't depend on
    # a specific font name; the registry orders them alphabetically.
    list_response = await client.get("/fonts", headers=auth_headers)
    assert list_response.status_code == 200

    fonts = list_response.json()["fonts"]
    assert fonts, "expected at least one bundled system font for this test"

    first_font = fonts[0]
    assert first_font.get("scope") == "system"

    file_response = await client.get(
        f"/fonts/{first_font['name']}",
        headers=auth_headers,
    )

    assert file_response.status_code == 200
    assert file_response.headers["content-type"].startswith("font/")
    assert len(file_response.content) > 0


@pytest.mark.asyncio
async def test_fonts_upload_rejects_unsupported_extension(
    client, db_session, auth_headers
):
    from tests.fixtures.factories import create_user

    await create_user(db_session, user_id="user-1", email="u@example.com")

    response = await client.post(
        "/fonts/upload",
        headers=auth_headers,
        files={"file": ("not-a-font.exe", b"nope", "application/octet-stream")},
    )

    # Unsupported extensions return 400, never 500, even though the handler
    # historically caught HTTPException and turned it into 500.
    assert response.status_code == 400