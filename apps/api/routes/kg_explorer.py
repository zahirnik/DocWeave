from fastapi import APIRouter, Response
router=APIRouter()
@router.get('/ui/kg')
async def ui_kg():
    return Response('KG Explorer installed', media_type='text/plain')
