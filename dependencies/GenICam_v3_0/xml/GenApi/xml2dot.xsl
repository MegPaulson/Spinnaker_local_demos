<?xml version="1.0" encoding="UTF-8"?>
<!-- ***************************************************************************
*  (c) 2010 by Basler Vision Technologies
*  Section: Vision Components
*  Project: GenApi
*  Author:  Fritz Dierks
******************************************************************************** -->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >

	<xsl:output method="text" encoding="UTF-8"/>
	<xsl:template match="/">
		//-----------------------------------------------------------------------------
		//  (c) 2010 by Basler Vision Technologies
		//  Section: Vision Components
		//  Project: GenApi
		//	Author:  Fritz Dierks
		//-----------------------------------------------------------------------------
		/*!
			\file     <xsl:value-of select="//Root/@Name"/>.dot
		*/
		//-----------------------------------------------------------------------------
		//  This file is generated automatically
		//  Do not modify!
		//-----------------------------------------------------------------------------

		digraph GenApi 
		{
			<xsl:apply-templates select="/*/*" mode="ListNodes"/>
			<xsl:apply-templates select="/*/*" mode="Connections"/>
		}	
	</xsl:template>

	<!-- ListNodes ******************************************** -->

	<xsl:template  match="*" mode="ListNodes">
      <xsl:choose>
          <xsl:when test="name()='Group'">
            <xsl:apply-templates select="./*" mode="ListNodes"/>
          </xsl:when>
			<xsl:when test="name()='StructReg'">subgraph cluster<xsl:value-of select="position()"/> {
				<xsl:apply-templates select="./*" mode="ListNodesStructReg"/>
				label="StructReg::<xsl:value-of select="@Comment"/>";
				graph[style=dotted];
			}
          </xsl:when>
			<xsl:otherwise>
		<xsl:value-of select="@Name"/> [label="<xsl:value-of select="name()"/>::<xsl:value-of select="@Name"/>"];
			</xsl:otherwise>
      </xsl:choose>

	</xsl:template>
	<!-- ListNodesStructReg ******************************************** -->

	<xsl:template  match="*" mode="ListNodesStructReg">
				<xsl:if test="name()='StructEntry'"><xsl:value-of select="@Name"/> [label="<xsl:value-of select="name()"/>::<xsl:value-of select="@Name"/>"];</xsl:if>
  </xsl:template>
   
	<!-- Connections ******************************************** -->

	<xsl:template  match="*" mode="Connections">
      <xsl:choose>
			<xsl:when test="name()='Group'">
            <xsl:apply-templates select="./*" mode="Connections"/>
			</xsl:when>
			<xsl:when test="name()='StructReg'">
            <xsl:apply-templates select="./*" mode="InsideConnectionsStructReg"/>
			</xsl:when>
			<xsl:otherwise>
            <xsl:apply-templates select="./*" mode="InsideConnections"/>
			</xsl:otherwise>
      </xsl:choose>
	</xsl:template>

  <!-- InsideConnectionsStructReg ******************************************** -->

  <xsl:template  match="*" mode="InsideConnectionsStructReg">
    <xsl:if test="name()='StructEntry'">
      <xsl:variable name="CurentNode">
        <xsl:value-of select="@Name"/>
      </xsl:variable>
      <xsl:for-each select="../*">
			<xsl:if test="starts-with(name(), 'p') and not(name()='pTerminal') and not(name()='pDependent')">
          <xsl:value-of select="$CurentNode"/> -> <xsl:value-of select="."/> [label="<xsl:value-of select="name()"/>"];
			</xsl:if>
      </xsl:for-each>
    </xsl:if>
  </xsl:template>

  <!-- InsideConnections ******************************************** -->
  <xsl:template  match="*" mode="InsideConnections">
	    <!-- Show all links except those  created by the post-processor -->
		<xsl:if test="starts-with(name(), 'p') and not(name()='pTerminal') and not(name()='pDependent')">
			<xsl:value-of select="../@Name"/> -> <xsl:value-of select="."/> [label="<xsl:value-of select="name()"/>"];
		</xsl:if>
	</xsl:template>

</xsl:stylesheet>
